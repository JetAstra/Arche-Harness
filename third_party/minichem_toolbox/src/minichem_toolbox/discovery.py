"""Neutral, task-independent progressive discovery over the frozen catalog."""

from __future__ import annotations

import difflib
import re
from copy import deepcopy
from typing import Any, Literal, Mapping

from .action_aliases import aliases_for_action
from .catalog import (
    CATEGORY_LABELS,
    _public_action_text,
    _public_backend_text,
    _public_parameter_text,
    active_catalog_snapshot,
    action_specs,
    backend_specs,
)
from .models import ActionSpec, BackendSpec
from .parameter_specs import (
    RESOURCE_LIMIT_PARAMETER_SPECS,
    common_fixed_parameter_specs,
    inferred_parameter_metadata,
)
from .resource_budget import gaussian_resource_policy_record, resource_budget_record
from .search_index import BM25Index, normalize_scores, tokenize, weighted_text
from .semantic_embeddings import MODEL_ID, semantic_scores
from .timeout_policy import timeout_policy_record


ActionKind = Literal["all", "scientific", "data"]


_PUBLIC_RESOURCE_BUDGET = {
    "source": "service_policy",
    "agent_controllable": False,
    "scope": "per_task",
}

_PUBLIC_DEFAULT_SUPPRESSION = {
    ("sdf_to_gjf", "inputs.route_parameters"),
    ("xyz_to_gjf", "inputs.route_section"),
}


def _public_timeout_policy(execution_class: str) -> dict[str, Any]:
    """返回调用方需要的超时类别和边界，不公开配置来源名称。"""

    value = timeout_policy_record(execution_class)  # type: ignore[arg-type]
    value["source"] = "service_policy"
    return value


_COMPOSITE_CALCULATOR_ACTIONS = {
    "geometric": ("calculate_energy", "calculate_forces"),
    "sella": ("calculate_energy", "calculate_forces"),
}

_ACTION_INPUT_HANDOFF_NOTES = {
    "generate_conformer_ensemble": (
        "inputs.molecule 可以直接使用 SMILES 字符串或 {'smiles': '<SMILES>'} 映射；"
        "选择 rdkit_etkdg 时该输入由 Action 自身生成三维构象，不需要先调用 SMILES/SDF 转换器。"
        "rdkit_etkdg 只嵌入 RDKit 可解析的分子图，不能建立金属配位或离子接触；"
        "这类多接触结构使用 build_reactive_complex_ensemble，并显式给出 contacts、charge 和 multiplicity。"
        "选择 crest 时仍必须提供带坐标的 molecule 或 initial_structure；CREST 不接受裸 SMILES。"
    ),
    "smiles2sdf": (
        "inputs.smiles_list 是该转换 Action 的分子列表。primary 输出是包含成功记录的聚合 "
        "SDFFile；每个成功项目另有一个单记录 SDFRecordFile ArtifactRef，位于 "
        "result.successful_records[*].sdf_artifact。items 和 messages 保留全部输入及失败原因；"
        "只有 successful_records 中的成功项目可以逐项交给 sdf_to_gjf，失败项目不会进入下游。"
    ),
    "sdf_to_gjf": (
        "inputs.sdf_path 与 inputs.route_parameters 是该转换 Action 的输入；"
        "route_parameters 提供显式 Gaussian route 参数。"
        "该 Action 只生成并预检 GJF，不运行 Gaussian。"
    ),
    "xyz_to_gjf": (
        "inputs.input_xyz_path 与 inputs.route_section 是该转换 Action 的输入；"
        "input_xyz_path 同时接受 XYZFile 和 AtomicStructureFile ArtifactRef；"
        "route_section 提供显式 Gaussian route 内容。"
        "该 Action 只生成并预检 GJF，不运行 Gaussian。"
    ),
    "extract_final_structure": (
        "可选 inputs.reference_structure 接受 AtomicStructure 或 AtomicStructureFile ArtifactRef；"
        "它必须与 cclib 最后一帧逐项匹配原子数、元素顺序、电荷和多重度。匹配后按同一"
        "原子索引复制 atom_map_number，不进行图同构猜测或原子重排；完整结构保存在 primary"
        " AtomicStructure Artifact 中。"
    ),
    "generate_gaussian_code": (
        "该 Action 是一个可选、非默认的人工编制四阶段 route 组合小型 workflow，用于暂时缩小模型在 "
        "Gaussian 关键词上的探索范围；四次模型调用固定使用 DeepSeek Pro。它只接受 "
        "inputs.user_input 中的自然语言 Gaussian 计算需求，返回一行经过 route 文本和安全 "
        "形状检查的 Gaussian route；不做科学验证，不生成完整输入文件、不接收 structure、"
        "不提交或执行 Gaussian，也不产生科学证据。"
    ),
    "derive_vibrational_modes": (
        "inputs.hessian expects the primary Hessian ArtifactRef returned by "
        "calculate_hessian. inputs.structure separately expects the exact matching "
        "AtomicStructure returned in calculate_hessian.result.structure; this is "
        "especially important for Gaussian, whose formatted-checkpoint Hessian may be "
        "in a rotated standard orientation. Never reuse one artifact_id for both fields."
    ),
    "derive_thermochemistry": (
        "For backend_id=internal_thermochemistry, inputs.energy expects an EnergyResult "
        "ArtifactRef and inputs.frequencies expects the FrequencyResult ArtifactRef returned "
        "by derive_vibrational_modes. A calculate_hessian primary artifact is a Hessian, not "
        "a FrequencyResult; explicitly run the separate vibrational-mode Action first."
    ),
}

_SMILES2SDF_RESULT_FIELDS = [
    "input_count",
    "molecules_written",
    "failed_item_count",
    "output_sdf_path",
    "messages",
    "items",
    "successful_records",
]

_STRUCTURE_BATCH_ITEM_ARTIFACT_SCHEMA = {
    "type": ["object", "null"],
    "artifact_type": "ArtifactRef",
    "semantic_type": "XYZFile",
    "description": "紧凑 ArtifactRef；其路径承载该项实际坐标，可直接交给 xyz_to_gjf。",
}

_REACTIVE_COMPLEX_RESULT_FIELDS = [
    "ensemble",
    "rejected_candidates",
    "attempts",
    "count",
    "attempted_count",
    "accepted_count",
    "rejected_count",
]

_REACTIVE_COMPLEX_RESULT_SCHEMA = {
    "type": "object",
    "required": ["ensemble"],
    "properties": {
        "ensemble": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["candidate_id", "item_id", "structure_artifact"],
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "pattern": r"^candidate_[0-9]{4}$",
                    },
                    "item_id": {"type": "string"},
                    "structure_artifact": _STRUCTURE_BATCH_ITEM_ARTIFACT_SCHEMA,
                },
            },
        },
    },
}

_CONFORMER_ENSEMBLE_RESULT_FIELDS = [
    "ensemble",
    "ensemble_file",
    "count",
    "generation_settings",
    "energies",
    "weights",
]

_CONFORMER_ENSEMBLE_RESULT_SCHEMA = {
    "type": "object",
    "required": ["ensemble"],
    "properties": {
        "ensemble": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["conformer_id", "item_id", "structure_artifact"],
                "properties": {
                    "conformer_id": {"type": "string"},
                    "item_id": {"type": "string"},
                    "structure_artifact": _STRUCTURE_BATCH_ITEM_ARTIFACT_SCHEMA,
                },
            },
        },
        "ensemble_file": {"type": "string"},
    },
}

_SUBSTRUCTURE_MATCH_RESULT_FIELDS = [
    "query",
    "query_format",
    "matches",
    "match_count",
]

_SUBSTRUCTURE_MATCH_RESULT_SCHEMA = {
    "type": "object",
    "required": _SUBSTRUCTURE_MATCH_RESULT_FIELDS,
    "properties": {
        "query": {"type": "string"},
        "query_format": {"type": "string", "enum": ["smarts", "smiles"]},
        "matches": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "integer", "minimum": 0},
            },
        },
        "match_count": {"type": "integer", "minimum": 0},
    },
}

_STRUCTURE_BATCH_HANDOFF = {
    "source_field": "/result/ensemble",
    "item_field": "/structure_artifact",
    "item_id_field": "/item_id",
    "target_field": "/inputs/input_xyz_path",
    "downstream_action_id": "xyz_to_gjf",
    "max_items": 32,
}

_SMILES2SDF_RESULT_SCHEMA = {
    "type": "object",
    "required": [
        "input_count",
        "molecules_written",
        "failed_item_count",
        "messages",
        "items",
        "successful_records",
    ],
    "properties": {
        "input_count": {
            "type": "integer",
            "minimum": 1,
            "maximum": 32,
        },
        "molecules_written": {
            "type": "integer",
            "minimum": 0,
            "maximum": 32,
        },
        "failed_item_count": {
            "type": "integer",
            "minimum": 0,
            "maximum": 32,
        },
        "output_sdf_path": {
            "type": "string",
            "description": "成功或部分成功时为聚合 SDFFile 的工作区相对路径；全失败时省略。",
        },
        "success": {
            "type": "boolean",
            "description": "有成功记录时保留的兼容状态字段；全失败结果可以省略。",
        },
        "messages": {
            "type": "array",
            "items": {"type": "string"},
        },
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "items": {
                "type": "object",
                "required": ["item_id", "input_index", "smiles", "status"],
                "properties": {
                    "item_id": {"type": "string"},
                    "input_index": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 31,
                    },
                    "smiles": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["success", "failed"],
                    },
                    "error": {
                        "type": ["object", "null"],
                    },
                },
            },
        },
        "successful_records": {
            "type": "array",
            "maxItems": 32,
            "items": {
                "type": "object",
                "required": ["item_id", "input_index", "sdf_artifact"],
                "properties": {
                    "item_id": {"type": "string"},
                    "input_index": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 31,
                    },
                    "sdf_artifact": {
                        "type": "object",
                        "artifact_type": "ArtifactRef",
                        "semantic_type": "SDFRecordFile",
                        "description": "紧凑 ArtifactRef；其 semantic_type 固定为 SDFRecordFile。",
                    },
                },
            },
        },
    },
}

_SMILES2SDF_BATCH_HANDOFF = {
    "source_field": "/result/successful_records",
    "item_field": "/sdf_artifact",
    "item_id_field": "/item_id",
    "target_field": "/inputs/sdf_path",
    "max_items": 32,
    "downstream_action_id": "sdf_to_gjf",
    "only_successful_items": True,
    "description": (
        "只展开 successful_records 中的成功项目；items 中失败（status=failed）的输入保留在结果中，"
        "不会传给 sdf_to_gjf。每个 sdf_artifact 是单记录 SDFRecordFile，不是聚合 primary SDFFile。"
    ),
}

_ARTIFACT_REF_SCHEMA = {
    "type": "object",
    "artifact_type": "ArtifactRef",
    "required": ["artifact_id"],
    "properties": {
        "artifact_id": {"type": "string"},
    },
}

_ARCHE_TS_RESULT_FIELDS = [
    "workflow_id",
    "endpoint_pairs",
    "endpoint_pair_count",
    "path_image_candidates",
    "candidate_pagination",
    "stage_statuses",
    "failure_summary",
    "stage_diagnostics",
    "transition_states_validated",
]

_ARCHE_TS_PROVENANCE_SCHEMA = {
    "type": ["object", "null"],
    "description": (
        "ASH/KNARR 嵌套阶段的无路径来源记录；只保留 backend、模块、协议、"
        "版本和选择规则，供下游核对计算来源。"
    ),
    "required": [
        "schema_version",
        "backend_id",
        "backend_version",
        "module",
        "module_version",
        "protocol",
        "protocol_version",
        "selection_rule_version",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "backend_id": {"type": "string", "const": "ash_neb_uma"},
        "backend_version": {"type": ["string", "null"]},
        "module": {"type": "string", "const": "minichem_toolbox.backends.ash_neb"},
        "module_version": {"type": ["string", "null"]},
        "protocol": {"type": "string", "const": "search_reaction_path/ash_neb_uma"},
        "protocol_version": {"type": "string"},
        "selection_rule_version": {"type": ["string", "null"]},
        "engine": {"type": "string"},
        "calculator": {"type": "string"},
        "path_identity_metadata_restored": {"type": "boolean"},
        "path_is_validated_transition_state": {"type": "boolean"},
        "endpoint_atom_order_checked": {"type": "boolean"},
        "endpoint_atom_maps_checked": {"type": "boolean"},
    },
}

_ARCHE_TS_PAIR_SCHEMA = {
    "type": "object",
    "required": [
        "pair_id",
        "reactant_artifact",
        "product_artifact",
        "endpoint_artifact_source",
        "reaction_path_artifact",
        "transition_state_guess_artifact",
        "transition_state_guess_source",
        "transition_state_guess_source_image_index",
        "transition_state_guess_selection_rule_version",
        "path_image_artifacts",
        "stage_statuses",
        "provenance",
    ],
    "properties": {
        "pair_id": {"type": "string"},
        "reactant_artifact": {**_ARTIFACT_REF_SCHEMA, "type": ["object", "null"]},
        "product_artifact": {**_ARTIFACT_REF_SCHEMA, "type": ["object", "null"]},
        "reaction_path_artifact": {**_ARTIFACT_REF_SCHEMA, "type": ["object", "null"]},
        "transition_state_guess_artifact": {
            **_ARTIFACT_REF_SCHEMA,
            "type": ["object", "null"],
            "description": (
                "未验证过渡态初猜的 XYZFile ArtifactRef；可直接绑定 "
                "xyz_to_gjf.inputs.input_xyz_path。"
            ),
        },
        "transition_state_guess_source": {
            "type": ["string", "null"],
            "enum": [
                "knarr_saddle_xyz",
                "ash_saddle_geometry",
                "highest_interior_path_image",
                None,
            ],
            "description": "未验证初猜的机器可读来源；null 表示没有可交接初猜。",
        },
        "transition_state_guess_source_image_index": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "来源为路径图像时的零起始图像索引；文件或无初猜时为 null。",
        },
        "transition_state_guess_selection_rule_version": {
            "type": ["string", "null"],
            "description": "选择未验证初猜所使用的规则版本；未提供来源时为 null。",
        },
        "endpoint_artifact_source": {
            "type": "string",
            "enum": ["uma", "mmff_prepared"],
        },
        "stage_statuses": {"type": "object"},
        "provenance": _ARCHE_TS_PROVENANCE_SCHEMA,
        "path_image_artifacts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["candidate_id", "image_index"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "image_index": {"type": "integer", "minimum": 0},
                },
            },
        },
    },
}

_ARCHE_TS_PATH_CANDIDATE_SCHEMA = {
    "type": "object",
    "required": ["candidate_id", "pair_id", "image_index", "xyz_artifact"],
    "properties": {
        "candidate_id": {"type": "string"},
        "pair_id": {"type": "string"},
        "image_index": {"type": "integer", "minimum": 0},
        "xyz_artifact": {**_ARTIFACT_REF_SCHEMA},
    },
}

_ARCHE_TS_CANDIDATE_PAGINATION_SCHEMA = {
    "type": "object",
    "required": [
        "artifact",
        "read_tool",
        "json_pointer",
        "item_pointer_template",
        "continuation_content_format",
        "page_size",
        "item_start",
        "items_returned",
        "total_items",
        "continuation",
    ],
    "properties": {
        "artifact": {"type": ["object", "null"], "artifact_type": "ArtifactRef"},
        "read_tool": {"type": "string", "const": "read_workspace_text"},
        "json_pointer": {
            "type": "string",
            "const": "/path_image_candidates",
        },
        "item_pointer_template": {
            "type": "string",
            "const": "/path_image_candidates/{index}",
        },
        "continuation_content_format": {
            "type": "string",
            "const": "json_array_member_sequence",
            "description": (
                "continuation 读取到的是按候选边界切开的 JSON 数组成员序列，不含外层方括号；"
                "需要单个候选时，优先按 item_pointer_template 读取完整 JSON 对象。"
            ),
        },
        "page_size": {"type": "integer", "minimum": 32},
        "item_start": {"type": "integer", "minimum": 0},
        "items_returned": {"type": "integer", "minimum": 0},
        "total_items": {"type": "integer", "minimum": 0},
        "continuation": {
            "type": ["object", "null"],
            "properties": {
                "item_start": {"type": "integer", "minimum": 0},
                "item_count": {"type": "integer", "minimum": 1},
                "offset": {"type": "integer", "minimum": 0},
                "max_chars": {"type": "integer", "minimum": 1},
                "next_offset": {"type": ["integer", "null"], "minimum": 0},
            },
        },
    },
}

_ARCHE_TS_RESULT_SCHEMA = {
    "type": "object",
    "required": [
        "workflow_id",
        "endpoint_pairs",
        "endpoint_pair_count",
        "path_image_candidates",
        "stage_statuses",
        "stage_diagnostics",
        "transition_states_validated",
    ],
    "properties": {
        "workflow_id": {"type": "string"},
        "endpoint_pairs": {
            "type": "array",
            "items": _ARCHE_TS_PAIR_SCHEMA,
        },
        "endpoint_pair_count": {"type": "integer", "minimum": 0},
        "path_image_candidates": {
            "type": "array",
            "description": (
                "公开结果中的全部路径图像候选；结果不得截断。若调用方需要分段读取，"
                "可按 candidate_pagination 提供的指针继续读取。"
            ),
            "items": _ARCHE_TS_PATH_CANDIDATE_SCHEMA,
        },
        "candidate_pagination": _ARCHE_TS_CANDIDATE_PAGINATION_SCHEMA,
        "stage_statuses": {"type": "object"},
        "failure_summary": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": [
                    "pair_id",
                    "stage",
                    "status",
                    "reason_code",
                    "retryable",
                ],
                "properties": {
                    "pair_id": {"type": "string"},
                    "stage": {"type": "string"},
                    "status": {"type": "string", "enum": ["failed", "skipped"]},
                    "reason_code": {
                        "type": "string",
                        "description": (
                            "固定阶段的安全错误分类；未知内部错误使用泛化的 "
                            "pair_stage_failed 或 pair_stage_skipped。"
                        ),
                    },
                    "retryable": {
                        "type": "boolean",
                        "description": "该阶段失败是否允许上层重试整个固定 Action。",
                    },
                    "message": {
                        "type": "string",
                        "description": "嵌套 ASH 阶段提供的已脱敏、可行动错误说明。",
                    },
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "嵌套 ASH 阶段提供的已脱敏修正提示。",
                    },
                },
            },
        },
        "stage_diagnostics": {
            "type": ["array", "null"],
            "description": (
                "嵌套 ASH/KNARR partial 或失败的可行动 message/warnings；已去除路径、"
                "异常原文和无关敏感字段。"
            ),
            "items": {
                "type": "object",
                "required": ["pair_id", "stage", "status"],
                "properties": {
                    "pair_id": {"type": "string"},
                    "stage": {"type": "string"},
                    "status": {"type": "string", "enum": ["partial", "failed"]},
                    "message": {"type": "string"},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "transition_states_validated": {
            "type": "boolean",
            "const": False,
            "description": (
                "该 TS Pipeline 固定返回 false，表示它只生成未验证初猜；初猜不能作为"
                "过渡态证据。后续独立验证产生新的结果和 Artifact，不会回写此字段。"
            ),
        },
    },
}

_ARCHE_TS_BATCH_HANDOFF = {
    "source_field": "/result/endpoint_pairs",
    "item_field": "/transition_state_guess_artifact",
    "item_id_field": "/pair_id",
    "target_field": "/inputs/input_xyz_path",
    "max_items": 32,
    "downstream_action_id": "xyz_to_gjf",
    "only_present_items": True,
    "description": (
        "只展开 endpoint_pairs 中存在 transition_state_guess_artifact 的项目；字段为 null "
        "或缺失时跳过。该 XYZFile Artifact 可直接绑定 xyz_to_gjf.inputs.input_xyz_path，"
        "但仍必须独立完成 Gaussian TS 优化、频率检查和需要时的双向 IRC。"
    ),
}


_ATOMIC_STRUCTURE_RESULT_SCHEMA = {
    "type": "object",
    "required": ["atoms", "charge", "multiplicity", "pbc"],
    "properties": {
        "atoms": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["element", "position_angstrom"],
                "properties": {
                    "element": {"type": "string"},
                    "atomic_number": {"type": "integer", "minimum": 1},
                    "position_angstrom": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "formal_charge": {"type": "integer"},
                    "radical_electrons": {"type": "integer", "minimum": 0},
                    "atom_map_number": {"type": "integer", "minimum": 1},
                },
            },
        },
        "charge": {"type": "integer"},
        "multiplicity": {"type": "integer", "minimum": 1},
        "pbc": {
            "type": "array",
            "items": {"type": "boolean"},
            "minItems": 3,
            "maxItems": 3,
        },
        "metadata": {"type": "object"},
        "atom_map_numbers": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
    },
}


_PATH_IMAGE_EXTRACTOR_RESULT_SCHEMA = {
    "type": "object",
    "required": [
        "structure",
        "candidate_id",
        "pair_id",
        "image_index",
        "image_count",
        "transition_state_validated",
    ],
    "properties": {
        "structure": {"type": "object"},
        "candidate_id": {"type": "string"},
        "pair_id": {"type": "string"},
        "image_index": {"type": "integer", "minimum": 0},
        "image_count": {"type": "integer", "minimum": 1},
        "transition_state_validated": {"type": "boolean"},
    },
}

_PATH_IMAGE_EXTRACTOR_RESULT_FIELDS = [
    "structure",
    "candidate_id",
    "pair_id",
    "image_index",
    "image_count",
    "transition_state_validated",
]

_EXTRACT_FINAL_STRUCTURE_RESULT_FIELDS = [
    "structure",
    "coordinate_frame_index",
    "coordinate_frame_count",
    "normal_termination",
    "source_path",
    "require_normal_termination",
    "identity_source",
]

_EXTRACT_FINAL_STRUCTURE_RESULT_SCHEMA = {
    "type": "object",
    "required": [
        "structure",
        "coordinate_frame_index",
        "coordinate_frame_count",
        "normal_termination",
        "source_path",
        "require_normal_termination",
    ],
    "properties": {
        "structure": _ATOMIC_STRUCTURE_RESULT_SCHEMA,
        "coordinate_frame_index": {"type": "integer", "minimum": 0},
        "coordinate_frame_count": {"type": "integer", "minimum": 1},
        "normal_termination": {"type": "boolean"},
        "source_path": {"type": "string"},
        "require_normal_termination": {"type": "boolean"},
        "identity_source": {"type": "string"},
    },
}

_SCAN_REACTION_COORDINATE_RESULT_FIELDS = [
    "coordinate_type",
    "atom_indices",
    "coordinate_values",
    "coordinate_unit",
    "energies_hartree",
    "structures",
    "data_file",
    "completed_points",
    "requested_points",
]

_SCAN_REACTION_COORDINATE_RESULT_SCHEMA = {
    "type": "object",
    "required": _SCAN_REACTION_COORDINATE_RESULT_FIELDS,
    "properties": {
        "coordinate_type": {"type": "string"},
        "atom_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "coordinate_values": {
            "type": "array",
            "items": {"type": "number"},
        },
        "coordinate_unit": {"type": "string"},
        "energies_hartree": {
            "type": "array",
            "items": {"type": "number"},
        },
        "structures": {
            "type": "array",
            "items": _ATOMIC_STRUCTURE_RESULT_SCHEMA,
        },
        "data_file": {"type": ["string", "null"]},
        "completed_points": {"type": "integer", "minimum": 0},
        "requested_points": {"type": "integer", "minimum": 1},
    },
}

_PYSISYPHUS_REACTION_PATH_RESULT_FIELDS = [
    "path_file",
    "image_files",
    "image_count",
    "energies_hartree",
    "energy_source",
    "highest_energy_image_index",
    "highest_interior_energy_image_index",
    "highest_energy_image",
    "highest_interior_energy_image",
    "converged",
    "endpoints_consumed",
    "transition_state_validated",
]

_PYSISYPHUS_REACTION_PATH_RESULT_SCHEMA = {
    "type": "object",
    "required": _PYSISYPHUS_REACTION_PATH_RESULT_FIELDS,
    "properties": {
        "path_file": {"type": ["string", "null"]},
        "image_files": {"type": "array", "items": {"type": "string"}},
        "image_count": {"type": "integer", "minimum": 0},
        "energies_hartree": {
            "type": ["array", "null"],
            "items": {"type": "number"},
        },
        "energy_source": {"type": ["string", "null"]},
        "highest_energy_image_index": {"type": ["integer", "null"], "minimum": 0},
        "highest_interior_energy_image_index": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "highest_energy_image": {"type": ["object", "null"]},
        "highest_interior_energy_image": {"type": ["object", "null"]},
        "converged": {"type": "boolean"},
        "endpoints_consumed": {"type": "boolean"},
        "transition_state_validated": {"type": "boolean", "const": False},
    },
}

_ASH_REACTION_PATH_RESULT_FIELDS = [
    "images",
    "energies_hartree",
    "energy_image_indices",
    "energy_unit",
    "highest_energy_image_index",
    "highest_interior_image_index",
    "transition_state_guess",
    "diagnostic_ash_saddle_geometry",
    "interior_peak_candidate_valid",
    "converged",
    "energy_index_complete",
    "final_path_complete",
    "total_images",
    "intermediate_images_passed_to_ash",
    "climbing_image",
    "maximum_force_ev_per_angstrom",
    "rms_force_ev_per_angstrom",
    "maximum_steps",
    "interpolation",
    "model",
    "task_name",
    "device",
    "allow_model_download",
    "path_file",
    "initial_guess_path",
]

_ASH_REACTION_PATH_RESULT_SCHEMA = {
    "type": "object",
    "required": _ASH_REACTION_PATH_RESULT_FIELDS,
    "properties": {
        "images": {"type": "array", "items": {"type": "object"}},
        "energies_hartree": {"type": "array", "items": {"type": "number"}},
        "energy_image_indices": {"type": "array", "items": {"type": "integer"}},
        "energy_unit": {"type": "string", "const": "hartree"},
        "highest_energy_image_index": {"type": ["integer", "null"], "minimum": 0},
        "highest_interior_image_index": {"type": ["integer", "null"], "minimum": 0},
        "transition_state_guess": {"type": ["object", "null"]},
        "diagnostic_ash_saddle_geometry": {"type": ["object", "null"]},
        "interior_peak_candidate_valid": {"type": "boolean"},
        "converged": {"type": "boolean"},
        "energy_index_complete": {"type": "boolean"},
        "final_path_complete": {"type": "boolean"},
        "total_images": {"type": "integer", "minimum": 0},
        "intermediate_images_passed_to_ash": {"type": "integer", "minimum": 0},
        "climbing_image": {"type": "boolean"},
        "maximum_force_ev_per_angstrom": {"type": "number"},
        "rms_force_ev_per_angstrom": {"type": "number"},
        "maximum_steps": {"type": "integer", "minimum": 0},
        "interpolation": {"type": "string"},
        "model": {"type": "string"},
        "task_name": {"type": "string"},
        "device": {"type": "string"},
        "allow_model_download": {"type": "boolean"},
        "path_file": {"type": ["string", "null"]},
        "initial_guess_path": {"type": ["string", "null"]},
    },
}

_REACTION_COORDINATE_ANALYSIS_RESULT_FIELDS = [
    "image_count",
    "reaction_coordinate",
    "cumulative_aligned_rmsd_angstrom",
    "energies",
    "energy_unit",
    "relative_energies",
    "relative_energies_kcal_mol",
    "highest_energy_image_index",
    "highest_energy_relative_kcal_mol",
    "interior_local_maxima_indices",
    "interior_local_minima_indices",
    "bond_distance_profiles",
    "atom_maps_checked",
    "atom_identity_consistent",
    "atom_identity_basis",
    "transition_state_validated",
    "validation_note",
]

_REACTION_COORDINATE_ANALYSIS_RESULT_SCHEMA = {
    "type": "object",
    "required": _REACTION_COORDINATE_ANALYSIS_RESULT_FIELDS,
    "properties": {
        "image_count": {"type": "integer", "minimum": 1},
        "reaction_coordinate": {"type": "array", "items": {"type": "number"}},
        "cumulative_aligned_rmsd_angstrom": {
            "type": "array",
            "items": {"type": "number"},
        },
        "energies": {"type": "array", "items": {"type": "number"}},
        "energy_unit": {"type": "string"},
        "relative_energies": {"type": "array", "items": {"type": "number"}},
        "relative_energies_kcal_mol": {
            "type": "array",
            "items": {"type": "number"},
        },
        "highest_energy_image_index": {"type": "integer", "minimum": 0},
        "highest_energy_relative_kcal_mol": {"type": "number"},
        "interior_local_maxima_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "interior_local_minima_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "bond_distance_profiles": {"type": "array", "items": {"type": "object"}},
        "atom_maps_checked": {"type": "boolean"},
        "atom_identity_consistent": {"type": "boolean"},
        "atom_identity_basis": {"type": "string"},
        "transition_state_validated": {"type": "boolean", "const": False},
        "validation_note": {"type": "string"},
    },
}

_REACTION_PATH_IMAGE_SELECTION_HANDOFF = {
    "downstream_action_id": "extract_reaction_path_image",
    "source_field": None,
    "target_field": "/inputs/path_image_candidate/image_index",
    "description": (
        "只把显式公开的候选图像索引绑定到 extract_reaction_path_image；"
        "该图像仍是未验证候选，不能据此断言过渡态，也不按数组下标推断。"
    ),
}

_ACTION_OUTPUT_CONTRACT_BACKEND_SUPPLEMENTS = {
    "search_reaction_path": {
        "pysisyphus": {
            "result_fields": _PYSISYPHUS_REACTION_PATH_RESULT_FIELDS,
            "result_schema": _PYSISYPHUS_REACTION_PATH_RESULT_SCHEMA,
            "artifact_semantic_types": ["ReactionPath"],
            "output_note": (
                "pysisyphus 返回的路径、能量和最高内部图像只是未验证的 TS 候选；"
                "必须独立完成 TS 优化、频率检查和需要时的双向 IRC。"
            ),
            "image_selection_handoff": {
                **_REACTION_PATH_IMAGE_SELECTION_HANDOFF,
                "source_field": "/result/highest_interior_energy_image_index",
            },
        },
        "ash_neb_uma": {
            "result_fields": _ASH_REACTION_PATH_RESULT_FIELDS,
            "result_schema": _ASH_REACTION_PATH_RESULT_SCHEMA,
            "artifact_semantic_types": ["ReactionPath"],
            "output_note": (
                "ASH/KNARR 返回的路径、能量和最高内部图像只是未验证的 TS 候选；"
                "必须独立完成 TS 优化、频率检查和需要时的双向 IRC。"
            ),
            "image_selection_handoff": {
                **_REACTION_PATH_IMAGE_SELECTION_HANDOFF,
                "source_field": "/result/highest_interior_image_index",
            },
        },
    },
}

_ACTION_OUTPUT_CONTRACT_SUPPLEMENTS = {
    "build_reactive_complex_ensemble": {
        "result_fields": _REACTIVE_COMPLEX_RESULT_FIELDS,
        "result_schema": _REACTIVE_COMPLEX_RESULT_SCHEMA,
        "artifact_semantic_types": ["ReactiveComplexEnsemble", "XYZFile"],
        "batch_handoff": _STRUCTURE_BATCH_HANDOFF,
    },
    "generate_conformer_ensemble": {
        "result_fields": _CONFORMER_ENSEMBLE_RESULT_FIELDS,
        "result_schema": _CONFORMER_ENSEMBLE_RESULT_SCHEMA,
        "artifact_semantic_types": ["ConformerEnsemble", "XYZFile"],
        "batch_handoff": _STRUCTURE_BATCH_HANDOFF,
    },
    "search_local_substructures": {
        "result_fields": _SUBSTRUCTURE_MATCH_RESULT_FIELDS,
        "result_schema": _SUBSTRUCTURE_MATCH_RESULT_SCHEMA,
        "artifact_semantic_types": ["SubstructureMatchResult"],
        "output_note": (
            "matches 是按查询原子顺序排列的零起始原子索引二维数组；"
            "第一个匹配的完整索引数组使用 /result/matches/0；"
            "其中第一个查询原子使用 /result/matches/0/0。"
            "只有选择规则唯一且明确时才能绑定具体匹配；数组项不是带 atom_indices 字段的对象。"
        ),
    },
    "smiles2sdf": {
        "result_fields": _SMILES2SDF_RESULT_FIELDS,
        "result_schema": _SMILES2SDF_RESULT_SCHEMA,
        "batch_handoff": _SMILES2SDF_BATCH_HANDOFF,
        "artifact_handoff": (
            "单条 SDF 转换优先按唯一 artifact.semantic_type=SDFFile primary Artifact 绑定到 "
            "sdf_to_gjf.inputs.sdf_path；批量逐项转换只能复制公开 batch_handoff，不能按结果数组下标取单条记录。"
        ),
    },
    "search_transition_state_path": {
        "result_fields": _ARCHE_TS_RESULT_FIELDS,
        "result_schema": _ARCHE_TS_RESULT_SCHEMA,
        "artifact_semantic_types": [
            "TransitionStateGuessPipelineResult",
            "AtomicStructure",
            "ReactionPath",
            "XYZFile",
        ],
        "batch_handoff": _ARCHE_TS_BATCH_HANDOFF,
        "output_note": (
            "科学输出仅是未验证的 TS guess/path candidate；transition_states_validated 始终为 false。"
            "初猜只能作为后续计算的初始几何，不能作为过渡态身份、反应连通性或机理结论的科学证据，"
            "若要用某个初猜支持目标过渡态及其反应通道，必须由后续 Action 独立完成 Gaussian TS "
            "优化；Hessian/frequency（频率）分析必须确认恰好一个"
            "法向模式对应目标反应坐标的虚频，其余频率为实频；forward 和 reverse IRC（双向 IRC）"
            "必须分别收敛，"
            "并根据原子映射、键变化和必要的立体化学核对为预期反应物和产物侧。"
        ),
    },
    "extract_reaction_path_image": {
        "result_fields": _PATH_IMAGE_EXTRACTOR_RESULT_FIELDS,
        "result_schema": _PATH_IMAGE_EXTRACTOR_RESULT_SCHEMA,
        "artifact_semantic_types": ["AtomicStructure", "XYZFile"],
        "artifact_handoff": (
            "结果 primary 是 AtomicStructure；返回的 XYZFile ArtifactRef 可直接绑定到 "
            "xyz_to_gjf.inputs.input_xyz_path，image_index 只按请求显式提取，不作自动 TS 判断。"
        ),
    },
    "extract_final_structure": {
        "result_fields": _EXTRACT_FINAL_STRUCTURE_RESULT_FIELDS,
        "result_schema": _EXTRACT_FINAL_STRUCTURE_RESULT_SCHEMA,
        "artifact_semantic_types": ["AtomicStructure", "AtomicStructureFile"],
        "artifact_handoff": (
            "primary AtomicStructure Artifact 保存完整的最后一帧和 atom_map_numbers；"
            "AtomicStructureFile XYZ ArtifactRef 也可直接作为后续 reference_structure 输入，"
            "但文件本身不承载 atom_map_numbers；"
            "提供 reference_structure 时，result/provenance.identity_source 为 "
            "reference_structure。"
        ),
    },
    "scan_reaction_coordinates": {
        "result_fields": _SCAN_REACTION_COORDINATE_RESULT_FIELDS,
        "result_schema": _SCAN_REACTION_COORDINATE_RESULT_SCHEMA,
    },
    "analyze_reaction_coordinate": {
        "result_fields": _REACTION_COORDINATE_ANALYSIS_RESULT_FIELDS,
        "result_schema": _REACTION_COORDINATE_ANALYSIS_RESULT_SCHEMA,
        "artifact_semantic_types": ["ReactionCoordinateAnalysisResult"],
        "output_note": (
            "最高能量图像仅是未验证候选；必须独立完成 TS 优化、频率检查和需要时的双向 IRC。"
        ),
        "image_selection_handoff": {
            **_REACTION_PATH_IMAGE_SELECTION_HANDOFF,
            "source_field": "/result/highest_energy_image_index",
        },
    },
}

_ACTION_INPUT_CONTRACT_SUPPLEMENTS = {
    "select_reactive_complex_candidate": {
        "ensemble": {
            "type": "ReactiveComplexEnsemble | ArtifactRef",
            "accepted_forms": [
                "typed ReactiveComplexEnsemble ArtifactRef, compact {'artifact_id': 'art_...'}, or exact artifact-id string",
                "structured mapping containing a non-empty accepted-candidate ensemble",
            ],
            "accepted_artifact_semantic_types": ["ReactiveComplexEnsemble"],
            "rejected_artifact_semantic_types": [
                "ConformerEnsemble",
                "AtomicStructure",
            ],
            "rejected_forms": [
                "ConformerEnsemble or AtomicStructure ArtifactRef",
                "path to a JSON summary; pass the structured mapping or typed artifact instead",
            ],
            "cardinality": "one aggregate ReactiveComplexEnsemble with explicit candidate_id selection",
        }
    }
}

_ACTION_SETTING_CONTRACT_SUPPLEMENTS = {
    "select_reactive_complex_candidate": {
        "candidate_id": {
            "description": (
                "必须逐字匹配输入 ReactiveComplexEnsemble 中一个已接受候选的 candidate_id；"
                "稳定格式为 candidate_ 加四位十进制序号（例如 candidate_0001），不能填写数组位置或数字 0。"
            ),
            "pattern": r"^candidate_[0-9]{4}$",
        },
    },
}


def _lexical_stem(value: str) -> str:
    """Small deterministic morphology normalizer, not a relevance ranker."""

    for suffix in (
        "izations",
        "isations",
        "ization",
        "isation",
        "ations",
        "ation",
        "ments",
        "ment",
        "ing",
        "ed",
        "es",
        "s",
    ):
        if value.endswith(suffix) and len(value) - len(suffix) >= 5:
            return value[: -len(suffix)]
    return value


def _matches_all_terms(terms: list[str], haystack: str) -> bool:
    if not terms:
        return True
    words = re.findall(r"[a-z0-9]+", haystack.casefold())
    word_stems = [_lexical_stem(word) for word in words]
    for term in terms:
        if term in haystack:
            continue
        stem = _lexical_stem(term)
        if not any(
            word.startswith(stem)
            or stem.startswith(word_stem)
            and len(word_stem) >= 5
            for word, word_stem in zip(words, word_stems)
        ):
            return False
    return True


def _snapshot(value: dict[str, Any] | None = None) -> dict[str, Any]:
    return value if value is not None else active_catalog_snapshot()


def _catalog_version(current: Mapping[str, Any]) -> str:
    """读取 canonical catalog 的稳定版本标识。

    v7 canonical catalog 尚未单独声明 ``catalog_version`` 字段；其公开版本就是
    canonical snapshot 的 ``schema_version``。这里显式使用该冻结快照字段，不从
    discovery 请求、任意响应 schema 或任务上下文猜测版本。
    """

    value = current.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("canonical catalog 缺少有效 catalog_version")
    return str(value)


def _catalog_identity(current: Mapping[str, Any]) -> dict[str, Any]:
    """返回所有 discovery 成功响应共用的冻结 catalog 身份。"""

    return {
        "catalog_hash": current.get("catalog_hash"),
        "catalog_version": _catalog_version(current),
    }


def _backend_health(snapshot: dict[str, Any], backend_id: str) -> dict[str, Any]:
    for backend in snapshot.get("backends", []):
        if backend.get("id") == backend_id:
            return _public_health_projection(dict(backend.get("health") or {}))
    return {}


def _public_health_projection(value: Any) -> Any:
    """按健康结论白名单返回公开字段，不遍历内部探针对象。"""

    if not isinstance(value, dict):
        return {}
    projected: dict[str, Any] = {}
    for field_name in ("status", "available", "reason_code"):
        if value.get(field_name) is not None:
            projected[field_name] = value[field_name]
    resources = value.get("resources")
    if isinstance(resources, dict):
        public_resources: dict[str, dict[str, Any]] = {}
        for resource_id, resource in resources.items():
            if not isinstance(resource, dict):
                continue
            item = {
                field_name: resource[field_name]
                for field_name in ("available", "reason_code", "resource_id", "status", "version")
                if field_name in resource
            }
            if item:
                public_resources[str(resource_id)] = item
        if public_resources:
            projected["resources"] = public_resources
    return projected


def _health_status(snapshot: dict[str, Any], backend_id: str) -> str:
    return str(_backend_health(snapshot, backend_id).get("status") or "not_probed")


def _resource_summary(resource: dict[str, Any]) -> dict[str, Any]:
    """返回资源选择所需的科学字段，不公开容量、数量或模型身份。"""

    return {
        "resource_id": resource.get("id"),
        "display_name": resource.get("display_name", resource.get("id")),
        "kind": resource.get("kind"),
        "version": resource.get("version"),
        "format": resource.get("format"),
        "available": bool(resource.get("available")),
        "selectable": bool(resource.get("selectable", True)),
        "compatible_backends": list(resource.get("compatible_backends") or []),
        "selection_syntax": resource.get("selection_syntax"),
    }


def _registered_resources(
    snapshot: dict[str, Any], backend_id: str
) -> list[dict[str, Any]]:
    return [
        _resource_summary(resource)
        for resource in snapshot.get("resources", [])
        if backend_id in (resource.get("compatible_backends") or [])
    ]


def _public_fixed_parameters(
    values: Mapping[str, Mapping[str, Any]],
    *,
    action_id: str | None = None,
) -> list[dict[str, Any]]:
    """只保留科学输入、方法和设置的固定参数说明。"""

    allowed_runtime_paths = {
        "backend_runtime.direction",
        "backend_runtime.charge",
        "backend_runtime.multiplicity",
    }
    public: list[dict[str, Any]] = []
    for field_path, metadata in values.items():
        if not field_path.startswith(("inputs.", "method_spec.", "action_settings.")) and field_path not in allowed_runtime_paths:
            continue
        if field_path == "method_spec.allow_model_download":
            continue
        item = {"path": field_path}
        for key in ("description", "reason", "allowed_values", "type", "required"):
            if key not in metadata:
                continue
            value = metadata[key]
            if key == "description" and action_id is not None:
                value, _impact = _public_parameter_text(
                    action_id,
                    field_path,
                    value,
                    metadata.get("impact"),
                )
            item[key] = value
        if len(item) > 1:
            public.append(item)
    return public


def _provider_contract(
    backend: BackendSpec,
    action_id: str,
    snapshot: dict[str, Any],
    *,
    detailed: bool,
) -> dict[str, Any]:
    """生成一个 Backend 的公开科学契约，不包含运行时或资源探针细节。"""

    fixed_parameters = common_fixed_parameter_specs(
        runtime=backend.runtime,
        executables=backend.executables,
        python_modules=backend.python_modules,
        resource_constraints=backend.resource_constraints,
        validation_level=backend.validation_levels.get(action_id),
    )
    fixed_parameters.update(backend.fixed_parameter_specs.get(action_id, {}))
    value: dict[str, Any] = {
        "backend_id": backend.id,
        "display_name": backend.display_name,
        "description": _public_backend_text(backend.id, backend.description),
        "health_status": _health_status(snapshot, backend.id),
        "required_input_fields": list(backend.required_input_fields.get(action_id, ())),
        "required_method_fields": list(backend.required_method_fields.get(action_id, ())),
        "required_setting_fields": list(backend.required_setting_fields.get(action_id, ())),
        "allowed_method_values": {
            field: list(choices)
            for field, choices in backend.allowed_method_values.get(action_id, {}).items()
        },
        "allowed_setting_values": {
            field: list(choices)
            for field, choices in backend.allowed_setting_values.get(action_id, {}).items()
        },
        "required_component_roles": list(
            backend.required_component_roles.get(action_id, ())
        ),
        "component_backend_options": {
            role: list(options)
            for role, options in backend.component_backend_options.get(action_id, {}).items()
        },
        "supported_system_types": list(
            backend.supported_system_types.get(action_id, ())
        ),
        "resource_constraints": {
            key: backend.resource_constraints[key]
            for key in ("minimum_gpu_count",)
            if key in backend.resource_constraints
        },
        "validation_level": (
            "registered_validation"
            if backend.validation_levels.get(action_id)
            else None
        ),
        "action_parameter_descriptions": {
            field_name: _public_parameter_text(
                action_id,
                (
                    "method_spec."
                    if field_name in backend.required_method_fields.get(action_id, ())
                    else "action_settings."
                )
                + field_name,
                backend.method_schema[field_name],
            )[0]
            for field_name in (
                *backend.required_method_fields.get(action_id, ()),
                *backend.required_setting_fields.get(action_id, ()),
            )
            if field_name in backend.method_schema
        },
        "backend_fixed_parameters": {
            item["path"]: {
                key: value
                for key, value in item.items()
                if key != "path"
            }
            for item in _public_fixed_parameters(
                fixed_parameters,
                action_id=action_id,
            )
        },
    }
    if detailed:
        value.update(
            {
                "health": _backend_health(snapshot, backend.id),
                "registered_resources": _registered_resources(snapshot, backend.id),
            }
        )
    return value


def _provider_summary(
    backend: BackendSpec, action_id: str, snapshot: dict[str, Any]
) -> dict[str, Any]:
    return {
        "backend_id": backend.id,
        "display_name": backend.display_name,
        "health_status": _health_status(snapshot, backend.id),
        "required_input_fields": list(backend.required_input_fields.get(action_id, ())),
        "required_method_fields": list(backend.required_method_fields.get(action_id, ())),
        "required_setting_fields": list(backend.required_setting_fields.get(action_id, ())),
        "required_component_roles": list(
            backend.required_component_roles.get(action_id, ())
        ),
        "supported_system_types": list(
            backend.supported_system_types.get(action_id, ())
        ),
        "validation_level": (
            "registered_validation"
            if backend.validation_levels.get(action_id)
            else None
        ),
    }


def _action_summary(specification: ActionSpec) -> dict[str, Any]:
    return {
        "id": specification.id,
        "category": specification.category,
        "description": _public_action_text(
            specification.id, "description", specification.description
        ),
        "primary_output": specification.primary_output,
        "backend_ids": list(specification.backend_ids),
        "required_inputs": list(specification.required_inputs),
        "optional_inputs": list(specification.optional_inputs),
        "selection_policy": specification.selection_policy,
        "data_action": specification.data_action,
        "batch_safe": specification.batch_safe,
    }


def _public_action_specification(specification: ActionSpec) -> dict[str, Any]:
    """返回完整 Action 结构中的公开科学字段和中性文案。"""

    value = specification.as_dict()
    value["description"] = _public_action_text(
        specification.id, "description", specification.description
    )
    value["input_description"] = _public_action_text(
        specification.id, "input_description", specification.input_description
    )
    return value


def _field_type(field_name: str, *, section: str) -> dict[str, Any]:
    """返回 progressive discovery 使用的紧凑机器可读字段形状提示。

    公开 Action dispatcher 保留三个可扩展映射，而不是为每个 Action 预先注册一个大型
    Pydantic 模型。这些提示明确所选 Action/Backend 合同，不会把所有 schema 提前加载到
    初始模型上下文，也不会替调用方选择科学数值。
    """

    structure_fields = {
        "structure",
        "initial_guess",
        "initial_structure",
        "molecule",
        "reactant",
        "product",
        "transition_state",
        "ground_state",
        "geometry",
        "reference",
        "reference_structure",
    }
    artifact_file_fields = {
        "control_file",
        "dos_file",
        "lobster_output",
        "output_file",
        "profile_definition_file",
        "second_order_force_constants_file",
        "structure_file",
        "third_order_force_constants_file",
    }
    mapping_fields = {
        "calculator_action_settings",
        "calculator_method",
        "chemical_species_mapping",
        "constraints",
        "flexible_parameters",
        "k_points",
        "label_groups",
        "molecule_counts",
        "pair_parameters",
        "pseudopotentials",
    }
    array_fields = {
        "atom_indices",
        "band_path",
        "bin_edges",
        "critical_point_types",
        "cutoffs_au",
        "fractions",
        "integrated_bond_list",
        "models",
        "output_files",
        "properties",
        "q_mesh",
        "q_path",
        "scf_thresholds",
        "selection",
        "state_pair",
        "temperatures_kelvin",
    }
    explicit_boolean_fields = {
        "internal_coordinates",
        "stability_analysis",
        "symmetry_correction",
    }
    explicit_integer_fields = {
        "steps_per_diagonalization",
    }
    explicit_numeric_fields = {
        "finite_difference_step",
        "initial_trust_radius",
        "minimum_bond_order",
        "minimum_model_quality",
        "step_length",
        "threshold",
        "trust_radius_angstrom",
    }
    boolean_prefixes = (
        "add_",
        "align",
        "allow_",
        "canonical_",
        "center_",
        "deduplicate_",
        "enforce_",
        "filter_",
        "fix_",
        "generate_",
        "idealize",
        "ignore_",
        "include_",
        "initialize_",
        "keep_",
        "largest_",
        "match_",
        "neutralize",
        "only_",
        "periodic",
        "prealign_",
        "project_",
        "reflect",
        "relax_",
        "reorder_",
        "replace_",
        "require_",
        "restart",
        "simplified",
        "soft_min",
        "strict_",
        "symmetry_correction",
        "three_point_",
        "unique",
        "update_",
        "use_",
        "write_",
    )
    integer_tokens = (
        "_count",
        "_cycles",
        "_interval",
        "_iterations",
        "_points",
        "_steps",
        "_stride",
        "_bands",
        "max_",
        "num_",
        "number_of_",
        "random_seed",
        "starting_atom_serial",
        "starting_residue_number",
    )
    numeric_tokens = (
        "_angstrom",
        "_bar",
        "_bohr",
        "_cm1",
        "_degrees",
        "_ev",
        "_fraction",
        "_fs",
        "_hartree",
        "_kelvin",
        "_kj_mol_nm",
        "_kcal_mol",
        "_micrometer",
        "_mol_l",
        "_pa",
        "_percent",
        "_seconds",
        "_threshold",
        "_tolerance",
        "_ry",
    )

    if section == "inputs" and field_name == "molecule":
        return {
            "type": "SMILES string | AtomicStructure | ArtifactRef | workspace-relative structure path",
            "canonical_inline_example": {
                "atoms": [
                    {
                        "element": "H",
                        "position_angstrom": [0.0, 0.0, 0.0],
                    }
                ],
                "charge": 0,
                "multiplicity": 1,
                "pbc": [False, False, False],
            },
            "accepted_forms": [
                "SMILES string",
                "{'smiles': '<SMILES>'} mapping",
                "full AtomicStructure mapping",
                "full immutable ArtifactRef",
                "compact {'artifact_id': 'art_...'}",
                "exact artifact-id string",
                "workspace-relative .xyz, .pdb, .sdf, .mol, or ASE-readable structure path",
            ],
            "note": (
                "所选 backend 可能施加更强的坐标要求；例如 CREST 必须使用带坐标的起始结构。"
            ),
        }

    if field_name in structure_fields:
        return {
            "type": "AtomicStructure | ArtifactRef | workspace-relative structure path",
            "canonical_inline_example": {
                "atoms": [
                    {
                        "element": "H",
                        "position_angstrom": [0.0, 0.0, 0.0],
                    }
                ],
                "charge": 0,
                "multiplicity": 1,
                "pbc": [False, False, False],
            },
            "accepted_forms": [
                "full AtomicStructure mapping",
                "full immutable ArtifactRef",
                "compact {'artifact_id': 'art_...'}",
                "exact artifact-id string",
                "workspace-relative .xyz, .pdb, .sdf, .mol, or ASE-readable structure path",
            ],
        }
    if section == "inputs" and field_name == "ensemble":
        return {
            "type": "ConformerEnsemble | ArtifactRef | workspace-relative conformer file",
            "accepted_forms": [
                "typed ConformerEnsemble ArtifactRef, compact artifact_id object, or exact artifact-id string",
                "structured mapping containing a non-empty ensemble or conformers list",
                "workspace-relative .sdf, .mol, or multi-frame .xyz file",
            ],
            "rejected_forms": [
                "path to a JSON summary; pass the structured JSON value or typed artifact instead",
                "single AtomicStructure when the Action requires multiple conformers",
            ],
            "cardinality": "multiple conformers with identical atom count and atom ordering",
        }
    if section == "inputs" and field_name == "wavefunction_file":
        return {
            "type": "ArtifactRef | workspace-relative wavefunction path",
            "accepted_extensions": [
                ".fch",
                ".fchk",
                ".wfn",
                ".wfx",
                ".mwfn",
                ".molden",
                ".47",
            ],
            "accepted_forms": [
                "full immutable ArtifactRef",
                "compact {'artifact_id': 'art_...'}",
                "exact artifact-id string",
                "workspace-relative wavefunction path with one accepted extension",
            ],
            "rejected_extensions": [".chk", ".cube"],
            "note": (
                "The selected Multiwfn adapter requires orbital/basis information; "
                "a binary Gaussian CHK must first be converted to FCHK, and a "
                "density-only cube is insufficient for this NCI/RDG Action."
            ),
        }
    if field_name in artifact_file_fields or field_name.endswith("_file"):
        return {
            "type": "ArtifactRef | workspace-relative path",
            "accepted_forms": [
                "full immutable ArtifactRef",
                "compact {'artifact_id': 'art_...'}",
                "exact artifact-id string",
                "workspace-relative input path when the selected adapter accepts files",
            ],
        }
    if field_name == "energy":
        return {"type": "EnergyResult | ArtifactRef"}
    if field_name == "frequencies":
        return {"type": "FrequencyResult | ArtifactRef"}
    if section == "inputs" and field_name == "free_energies":
        return {
            "type": "array[object] | LabeledFreeEnergyCollection ArtifactRef",
            "minimum_items": 2,
            "item_fields": [
                "label",
                "gibbs_free_energy",
                "optional conformer_id",
            ],
            "cardinality": (
                "at least two distinct labels; repeated labels are conformers; every "
                "supplied conformer_id is globally unique"
            ),
            "accepted_artifact_semantic_types": ["LabeledFreeEnergyCollection"],
            "rejected_artifact_semantic_types": [
                "SpectrumResult",
                "FrequencyResult",
                "FreeEnergySelectivityResult",
            ],
            "canonical_inline_example": [
                {
                    "label": "pathway_a",
                    "conformer_id": "a1",
                    "gibbs_free_energy": -100.0,
                },
                {
                    "label": "pathway_b",
                    "conformer_id": "b1",
                    "gibbs_free_energy": -99.997,
                },
            ],
        }
    if section == "inputs" and field_name == "cycle_states":
        return {
            "type": "array[object] | CatalyticCycleFreeEnergyProfile ArtifactRef",
            "minimum_items": 2,
            "item_fields": ["id", "state_type", "gibbs_free_energy"],
            "cardinality": (
                "equal numbers of globally unique intermediate and transition_state records "
                "in strict alternating cycle order, beginning with an intermediate"
            ),
            "accepted_artifact_semantic_types": ["CatalyticCycleFreeEnergyProfile"],
            "rejected_artifact_semantic_types": [
                "LabeledFreeEnergyCollection",
                "FreeEnergySelectivityResult",
                "ReactionPath",
            ],
            "canonical_inline_example": [
                {"id": "I0", "state_type": "intermediate", "gibbs_free_energy": 0.0},
                {"id": "TS1", "state_type": "transition_state", "gibbs_free_energy": 10.0},
                {"id": "I1", "state_type": "intermediate", "gibbs_free_energy": -5.0},
                {"id": "TS2", "state_type": "transition_state", "gibbs_free_energy": 8.0},
            ],
        }
    if section == "inputs" and field_name == "energy_unit":
        return {
            "type": "string",
            "allowed_values": ["hartree", "eV", "kcal_mol", "kJ_mol"],
        }
    if field_name == "hessian":
        return {
            "type": "Hessian | ArtifactRef",
            "accepted_forms": [
                "dense Hessian result mapping with matrix and unit",
                "primary immutable ArtifactRef with semantic_type='Hessian'",
                "compact {'artifact_id': 'art_...'} for that Hessian artifact",
                "exact Hessian artifact-id string",
            ],
        }
    if field_name == "electron_density":
        return {
            "type": "ElectronDensityResult ArtifactRef",
            "accepted_forms": [
                "primary immutable ArtifactRef with semantic_type='ElectronDensityResult' returned by calculate_correlated_electron_density",
                "compact {'artifact_id': 'art_...'} for that primary ElectronDensityResult artifact",
                "exact ElectronDensityResult artifact-id string",
            ],
            "rejected_forms": [
                "job.gbw or another path from result.files",
                "ElectronDensityWavefunction, BackendFile, or diagnostic artifact ids",
            ],
            "handoff_example": {
                "electron_density": "<copy output_artifacts item whose semantic_type is ElectronDensityResult, or its artifact_id>"
            },
        }
    if field_name == "density_file":
        return {
            "type": "ElectronDensityWavefunction | ElectronDensityGrid ArtifactRef | workspace-relative density file",
            "accepted_forms": [
                "output_artifacts item from export_electron_density_grid whose semantic_type is ElectronDensityWavefunction or ElectronDensityGrid",
                "compact {'artifact_id': 'art_...'} or exact artifact-id string for that typed artifact",
                "workspace-relative WFN, WFX, FCHK, MWFN, Molden, or cube path",
            ],
        }
    if field_name == "cutoffs_au":
        return {
            "type": "array",
            "items": {"type": "number"},
            "minimum_items": 1,
            "maximum_items": 100,
            "value_range": {"exclusive_minimum": 0.0, "exclusive_maximum": 0.1},
            "example": [0.001, 0.0015, 0.002],
            "compatibility_input": (
                "A comma-separated numeric string is normalized for compatibility, but new "
                "requests should always send a JSON array of numbers."
            ),
        }
    if field_name == "vibrations":
        return {"type": "FrequencyResult | ArtifactRef"}
    if section == "inputs" and field_name == "reaction":
        return {
            "type": "reaction SMILES string | {reactants: array, products: array}",
            "accepted_forms": [
                "atom-balanced reaction SMILES with exactly one >> separator",
                "mapping with explicit reactants and products molecule-SMILES arrays",
            ],
        }
    if section == "inputs" and field_name == "mapped_reaction":
        return {"type": "AtomMappedReaction | ArtifactRef"}
    if section == "inputs" and field_name == "reaction_changes":
        return {"type": "ReactionChangeSet | ArtifactRef"}
    if section == "inputs" and field_name == "components":
        return {
            "type": "array[AtomicStructure | ArtifactRef]",
            "minimum_items": 2,
            "maximum_items": 16,
        }
    if section == "inputs" and field_name == "contact_pairs":
        return {
            "type": "array[object]",
            "item_fields": [
                "component_a", "atom_a", "component_b", "atom_b",
                "target_distance_angstrom", "optional tolerance_angstrom",
            ],
            "indexing": "zero_based",
        }
    if section == "inputs" and field_name == "spectrum":
        return {"type": "SpectrumResult | ArtifactRef"}
    if section == "inputs" and field_name == "spectral_data":
        return {"type": "structured spectral lines | ParsedQuantumChemistryResult | ArtifactRef"}
    if section == "inputs" and field_name == "excited_states":
        return {"type": "ExcitedStateResult | ParsedQuantumChemistryResult | ArtifactRef"}
    if field_name in mapping_fields:
        return {"type": "object"}
    if field_name in array_fields or field_name.endswith("_files"):
        return {"type": "array"}
    if field_name in explicit_boolean_fields or field_name.startswith(boolean_prefixes):
        return {"type": "boolean"}
    if field_name in explicit_integer_fields or field_name in {
        "charge",
        "multiplicity",
        "spin",
        "order",
        "dimensions",
    }:
        return {"type": "integer"}
    if any(token in field_name for token in integer_tokens):
        return {"type": "integer"}
    if field_name in {"frequency_scale_factor", "zpe_scale_factor"}:
        return {"type": "number | documented symbolic value"}
    if field_name in explicit_numeric_fields:
        return {"type": "number"}
    if any(field_name.endswith(token) for token in numeric_tokens):
        return {"type": "number"}
    if section == "inputs" and field_name.endswith("constants"):
        return {"type": "array | ArtifactRef"}
    return {"type": "string | documented structured value"}


def _placeholder(
    field_name: str,
    *,
    section: str,
    choices: tuple[str, ...] = (),
    contract: Mapping[str, Any] | None = None,
) -> Any:
    """根据字段输入合同生成最小请求模板占位值。

    输入包括字段名、所在 section、可选的 choices 以及最终字段合同；函数先处理
    choices 和内置 ArtifactRef 兼容形状，再依据合同中的 canonical_inline_example、
    type、items 或 item_fields 生成标量、数组或对象模板。读取
    canonical_inline_example 时使用深拷贝，因此调用不会修改原字段合同。返回值保持
    合同声明的模板形状；没有可识别类型时回退为 ``<field_name>`` 占位字符串。
    """

    if choices:
        return f"<choose exactly one: {' | '.join(str(choice) for choice in choices)}>"
    if field_name == "hessian":
        return "<dense Hessian result mapping or primary Hessian artifact_id>"
    if field_name == "electron_density":
        return "<primary ElectronDensityResult artifact_id from calculate_correlated_electron_density>"
    if field_name == "density_file":
        return "<ElectronDensityWavefunction or ElectronDensityGrid artifact_id from export_electron_density_grid>"
    field_contract = (
        contract if contract is not None else _field_type(field_name, section=section)
    )
    inline_example = field_contract.get("canonical_inline_example")
    if inline_example is not None:
        return deepcopy(inline_example)

    shape = str(field_contract.get("type") or "")
    if shape == "boolean":
        return "<boolean>"
    if shape == "integer":
        return "<integer>"
    if shape == "number":
        return "<number>"
    if shape == "object":
        return {"<field>": "<value>"}
    array_shape = next(
        (
            part.strip()
            for part in shape.split("|")
            if part.strip().startswith("array")
        ),
        None,
    )
    if array_shape is not None:
        item_shape = ""
        if "[" in array_shape and array_shape.endswith("]"):
            item_shape = array_shape[array_shape.index("[") + 1 : -1].strip()
        item_spec = field_contract.get("items")
        if not item_shape and isinstance(item_spec, Mapping):
            item_shape = str(item_spec.get("type") or "").strip()
        if not item_shape and field_contract.get("item_fields"):
            item_shape = "object"
        if item_shape == "object":
            return [{"<field>": "<value>"}]
        if item_shape in {"boolean", "integer", "number", "string"}:
            return [f"<{item_shape}>"]
        return ["<value>"]
    if "AtomicStructure" in shape:
        return "<AtomicStructure, artifact_id, or workspace-relative structure path>"
    if "ArtifactRef" in shape:
        semantic_types = field_contract.get("accepted_artifact_semantic_types")
        if semantic_types:
            semantic_names = " or ".join(str(item) for item in semantic_types)
            return f"<{semantic_names} artifact_id>"
        return "<artifact_id or accepted workspace-relative path>"
    return f"<{field_name}>"


def _field_contracts(
    fields: tuple[str, ...] | list[str],
    *,
    section: str,
    required: bool,
    choices: Mapping[str, tuple[str, ...]] | None = None,
    reference: Mapping[str, str] | None = None,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    allowed = choices or {}
    descriptions = reference or {}
    details = metadata or {}
    contracts: list[dict[str, Any]] = []
    for field_name in fields:
        field_metadata = details.get(field_name, {})
        public_metadata = {
            key: value
            for key, value in field_metadata.items()
            if key
            in {
                "description",
                "impact",
                "type",
                "minimum",
                "maximum",
                "exclusive_minimum",
                "exclusive_maximum",
                "minimum_items",
                "maximum_items",
                "items",
                "allowed_values",
                "default",
                "deprecated",
                "replacement",
                "accepted_artifact_semantic_types",
            }
        }
        if field_name == "allow_model_download":
            public_metadata["description"] = "服务资源可用性选项。"
            public_metadata["impact"] = "改变资源准备方式，不改变所选科学方法。"
            public_metadata.pop("default", None)
        contract = {
            "name": field_name,
            "required": required,
            **_field_type(field_name, section=section),
            **inferred_parameter_metadata(
                section, field_name, required=required
            ),
            **(
                {"allowed_values": list(allowed[field_name])}
                if field_name in allowed
                else {}
            ),
            **(
                {"description": descriptions[field_name]}
                if field_name in descriptions
                else {}
            ),
            **public_metadata,
        }
        if required:
            # Required fields have no operational default because the dispatcher
            # rejects omission.  Some backend adapters keep defensive direct-call
            # fallbacks; those must not be presented as public Action defaults.
            contract.pop("default", None)
        contracts.append(contract)
    return contracts


def _uses_gaussian_service_policy(
    specification: ActionSpec,
    backend: BackendSpec,
) -> bool:
    """判断实际生成 Gaussian 输入的 Action 是否采用服务资源策略。"""

    return backend.id == "gaussian" or specification.primary_output == "GaussianInputDeck"


def _action_request_contract(
    specification: ActionSpec,
    backend: BackendSpec,
) -> dict[str, Any]:
    gaussian_service_managed = _uses_gaussian_service_policy(specification, backend)
    backend_inputs = backend.required_input_fields.get(specification.id, ())
    required_inputs = tuple(dict.fromkeys((*specification.required_inputs, *backend_inputs)))
    required_methods = backend.required_method_fields.get(specification.id, ())
    required_settings = backend.required_setting_fields.get(specification.id, ())
    method_choices = backend.allowed_method_values.get(specification.id, {})
    setting_choices = backend.allowed_setting_values.get(specification.id, {})
    registered_parameters = backend.parameter_specs.get(specification.id, {})
    registered_by_section: dict[str, dict[str, Mapping[str, Any]]] = {
        "inputs": {},
        "method_spec": {},
        "action_settings": {},
        "resource_limits": {},
    }
    for field_path, metadata in registered_parameters.items():
        section, field_name = field_path.split(".", 1)
        if section in registered_by_section:
            registered_by_section[section][field_name] = metadata
    optional_inputs = tuple(
        field_name
        for field_name in dict.fromkeys(
            (*specification.optional_inputs, *registered_by_section["inputs"])
        )
        if field_name not in required_inputs
    )
    optional_methods = tuple(
        field_name
        for field_name in dict.fromkeys(
            (
                *method_choices,
                *registered_by_section["method_spec"],
            )
        )
        if field_name not in required_methods
        and field_name not in required_settings
        and field_name not in setting_choices
        and field_name != "allow_model_download"
        and "conditional" not in field_name
    )
    optional_settings = tuple(
        field_name for field_name in dict.fromkeys(
            (*setting_choices, *registered_by_section["action_settings"])
        )
        if field_name not in required_settings
    )
    conditional_rules = []
    for field_name, description in backend.method_schema.items():
        if "conditional" not in field_name and " requires " not in f" {description.casefold()} ":
            continue
        public_rule = _public_parameter_text(
            specification.id,
            f"method_spec.{field_name}",
            description,
        )[0]
        conditional_rules.append({"name": field_name, "rule": public_rule})
    fixed_parameters = common_fixed_parameter_specs(
        runtime=backend.runtime,
        executables=backend.executables,
        python_modules=backend.python_modules,
        resource_constraints=backend.resource_constraints,
        validation_level=backend.validation_levels.get(specification.id),
    )
    fixed_parameters.update(backend.fixed_parameter_specs.get(specification.id, {}))
    policy_record = _public_timeout_policy(specification.execution_class)
    budget_record = resource_budget_record()

    resource_parameter_specs = (
        {}
        if gaussian_service_managed
        else {
            field_name: dict(metadata)
            for field_name, metadata in RESOURCE_LIMIT_PARAMETER_SPECS.items()
        }
    )
    if not gaussian_service_managed:
        for field_name, metadata in registered_by_section["resource_limits"].items():
            resource_parameter_specs.setdefault(field_name, {}).update(dict(metadata))
        for field_name in ("cpu_cores", "memory_mb", "gpu_count"):
            current_maximum = resource_parameter_specs[field_name].get("maximum")
            budget_maximum = int(budget_record[field_name])
            resource_parameter_specs[field_name]["maximum"] = (
                min(int(current_maximum), budget_maximum)
                if current_maximum is not None
                else budget_maximum
            )
            resource_parameter_specs[field_name]["evaluation_budget_reason"] = (
                "Single requests and the sum of concurrently active managed jobs cannot exceed "
                "the configured per-task resource budget."
            )
    resource_maximum_fields = (
        {} if gaussian_service_managed else {"cpu_cores": "maximum_cpu_cores"}
    )
    for field_name, constraint_name in resource_maximum_fields.items():
        maximum = backend.resource_constraints.get(constraint_name)
        if maximum is None:
            continue
        current_maximum = resource_parameter_specs[field_name].get("maximum")
        resource_parameter_specs[field_name]["maximum"] = (
            min(int(current_maximum), int(maximum))
            if current_maximum is not None
            else int(maximum)
        )
        reason = backend.resource_constraints.get("reason")
        if reason:
            resource_parameter_specs[field_name]["backend_limit_reason"] = str(reason)
    resource_minimum_fields = (
        {} if gaussian_service_managed else {"gpu_count": "minimum_gpu_count"}
    )
    for field_name, constraint_name in resource_minimum_fields.items():
        minimum = backend.resource_constraints.get(constraint_name)
        if minimum is None:
            continue
        current_minimum = resource_parameter_specs[field_name].get("minimum")
        resource_parameter_specs[field_name]["minimum"] = (
            max(int(current_minimum), int(minimum))
            if current_minimum is not None
            else int(minimum)
        )
        resource_parameter_specs[field_name]["default"] = max(
            int(resource_parameter_specs[field_name].get("default", 0)),
            int(minimum),
        )
        reason = backend.resource_constraints.get("reason")
        if reason:
            resource_parameter_specs[field_name]["backend_limit_reason"] = str(reason)

    sections = {
        "inputs": {
            "required": _field_contracts(
                required_inputs,
                section="inputs",
                required=True,
                metadata=registered_by_section["inputs"],
            ),
            "optional": _field_contracts(
                optional_inputs,
                section="inputs",
                required=False,
                metadata=registered_by_section["inputs"],
            ),
        },
        "method_spec": {
            "required": _field_contracts(
                required_methods,
                section="method_spec",
                required=True,
                choices=method_choices,
                reference=backend.method_schema,
                metadata=registered_by_section["method_spec"],
            ),
            "optional_documented": _field_contracts(
                optional_methods,
                section="method_spec",
                required=False,
                choices=method_choices,
                reference=backend.method_schema,
                metadata=registered_by_section["method_spec"],
            ),
            # 保留原发现字段名称，兼容已有客户端。
            "optional_with_enumerated_values": _field_contracts(
                optional_methods,
                section="method_spec",
                required=False,
                choices=method_choices,
                reference=backend.method_schema,
                metadata=registered_by_section["method_spec"],
            ),
        },
        "action_settings": {
            "required": _field_contracts(
                required_settings,
                section="action_settings",
                required=True,
                choices=setting_choices,
                reference=backend.method_schema,
                metadata=registered_by_section["action_settings"],
            ),
            "optional_documented": _field_contracts(
                optional_settings,
                section="action_settings",
                required=False,
                choices=setting_choices,
                reference=backend.method_schema,
                metadata=registered_by_section["action_settings"],
            ),
            "optional_with_enumerated_values": _field_contracts(
                optional_settings,
                section="action_settings",
                required=False,
                choices=setting_choices,
                reference=backend.method_schema,
                metadata=registered_by_section["action_settings"],
            ),
        },
        "resource_limits": (
            {"service_managed": gaussian_resource_policy_record()}
            if gaussian_service_managed
        else {
            "optional_with_defaults": [
                    {
                        "name": field_name,
                        "required": False,
                        "type": str(metadata.get("type") or "integer"),
                        "description": (
                            "Optional resource limit checked by the service policy."
                        ),
                    }
                    for field_name, metadata in resource_parameter_specs.items()
                ],
            }
        ),
    }
    for section_name in ("inputs", "method_spec", "action_settings"):
        for subsection_name in (
            "required",
            "optional",
            "optional_documented",
            "optional_with_enumerated_values",
        ):
            for item in sections[section_name].get(subsection_name, []):
                if "description" not in item:
                    continue
                field_path = f"{section_name}.{item['name']}"
                description, impact = _public_parameter_text(
                    specification.id,
                    field_path,
                    item.get("description"),
                    item.get("impact"),
                )
                item["description"] = description
                if "impact" in item and impact is not None:
                    item["impact"] = impact
                if (specification.id, field_path) in _PUBLIC_DEFAULT_SUPPRESSION:
                    item.pop("default", None)
    # 对同名字段施加 Action 专属合同，避免通用 ``ensemble`` 提示把
    # ReactiveComplexEnsemble 误报成 ConformerEnsemble；只修改发现输出，不放宽服务端校验。
    input_contract_supplements = _ACTION_INPUT_CONTRACT_SUPPLEMENTS.get(
        specification.id,
        {},
    )
    for section_name in ("required", "optional"):
        for item in sections["inputs"].get(section_name, []):
            supplement = input_contract_supplements.get(item["name"])
            if supplement:
                item.update(deepcopy(supplement))
    setting_contract_supplements = _ACTION_SETTING_CONTRACT_SUPPLEMENTS.get(
        specification.id,
        {},
    )
    for section_name in ("required", "optional_documented", "optional_with_enumerated_values"):
        for item in sections["action_settings"].get(section_name, []):
            supplement = setting_contract_supplements.get(item["name"])
            if supplement:
                item.update(deepcopy(supplement))
    required_contracts_by_section = {
        section_name: {
            item["name"]: item
            for item in section_value.get("required", [])
        }
        for section_name, section_value in sections.items()
    }

    template: dict[str, Any] = {"action_id": specification.id}
    if specification.selection_policy in {
        "agent_backend_required",
        "agent_components_required",
    }:
        template["backend_id"] = backend.id
    elif specification.selection_policy == "agent_source_required":
        template["source_id"] = backend.id
    component_roles = backend.required_component_roles.get(specification.id, ())
    if component_roles:
        template["component_backends"] = {
            role: f"<choose exactly one: {' | '.join(backend.component_backend_options[specification.id][role])}>"
            for role in component_roles
        }
    template["inputs"] = {
        field_name: _placeholder(
            field_name,
            section="inputs",
            contract=required_contracts_by_section["inputs"][field_name],
        )
        for field_name in required_inputs
    }
    template["method_spec"] = {
        field_name: _placeholder(
            field_name,
            section="method_spec",
            choices=method_choices.get(field_name, ()),
            contract=required_contracts_by_section["method_spec"][field_name],
        )
        for field_name in required_methods
    }
    template["action_settings"] = {
        field_name: _placeholder(
            field_name,
            section="action_settings",
            choices=setting_choices.get(field_name, ()),
            contract=required_contracts_by_section["action_settings"][field_name],
        )
        for field_name in required_settings
    }
    nested_component_actions = _COMPOSITE_CALCULATOR_ACTIONS.get(backend.id, ())
    if (
        nested_component_actions
        and "calculator_action_settings" in template["action_settings"]
    ):
        template["action_settings"]["calculator_action_settings"] = {
            action_id: {} for action_id in nested_component_actions
        }
    if not gaussian_service_managed:
        template["resource_limits"] = {
            field_name: None
            for field_name, metadata in resource_parameter_specs.items()
        }

    output_contract = {
        "primary_output": specification.primary_output,
        "result_envelope": "ActionResult",
        "artifact_handoff": (
            "Use each returned output_artifacts ArtifactRef, compact artifact_id object, or "
            "exact artifact-id string as the typed input to a later Action."
        ),
        "input_handoff_note": _ACTION_INPUT_HANDOFF_NOTES.get(specification.id),
    }
    output_contract.update(
        deepcopy(_ACTION_OUTPUT_CONTRACT_SUPPLEMENTS.get(specification.id, {}))
    )
    output_contract.update(
        deepcopy(
            _ACTION_OUTPUT_CONTRACT_BACKEND_SUPPLEMENTS.get(
                specification.id, {}
            ).get(backend.id, {})
        )
    )

    return {
        "template_kind": "fill_every_angle_bracket_before_execute_action",
        "execute_action_request_template": template,
        "sections": sections,
        "backend_fixed_parameters": _public_fixed_parameters(
            fixed_parameters,
            action_id=specification.id,
        ),
        "conditional_requirements": conditional_rules,
        "component_request_contracts": (
            {
                "calculator": {
                    "required_nested_actions": list(nested_component_actions),
                    "calculator_method_location": "method_spec.calculator_method",
                    "calculator_settings_location": (
                        "action_settings.calculator_action_settings.<nested_action>"
                    ),
                    "inspection_rule": (
                        "After choosing component_backends.calculator, inspect that exact "
                        "Backend for every required_nested_action and fill its required method "
                        "and setting fields. Empty nested mappings are valid only when the "
                        "selected calculator declares no settings for that nested Action."
                    ),
                }
            }
            if nested_component_actions
            else {}
        ),
        "output_contract": output_contract,
        "execution_timeout_policy": policy_record,
        **(
            {"resource_policy": gaussian_resource_policy_record()}
            if gaussian_service_managed
            else {"evaluation_resource_budget": dict(_PUBLIC_RESOURCE_BUDGET)}
        ),
        "execution_checklist": [
            "Replace every angle-bracket placeholder; placeholders are not defaults.",
            "Preserve the exact selected action_id and backend_id/source_id.",
            "Supply every required field in the section where it is listed.",
            "Apply every conditional requirement triggered by a caller-selected option.",
            "Match ArtifactRef.semantic_type to each input field; differently typed required inputs normally require different artifact ids.",
            (
                "Gaussian CPU and memory use service_managed_full_capacity and serial execution; "
                "the caller cannot override these resources."
                if gaussian_service_managed
                else "The service checks any supplied resource limits; this contract does not disclose service capacity."
            ),
        ],
    }


def _compact_action_request_contract(
    specification: ActionSpec,
    backend: BackendSpec,
) -> dict[str, Any]:
    """Return the bounded contract needed to execute one selected provider."""

    complete = _action_request_contract(specification, backend)
    sections = complete["sections"]

    def required(section: str) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in item.items()
                if key
                in {
                    "name",
                    "required",
                    "type",
                    "allowed_values",
                    "minimum",
                    "maximum",
                    "exclusive_minimum",
                    "exclusive_maximum",
                    "minimum_items",
                    "maximum_items",
                    "item_fields",
                    "accepted_forms",
                    "rejected_forms",
                    "accepted_artifact_semantic_types",
                    "rejected_artifact_semantic_types",
                    "cardinality",
                    "canonical_inline_example",
                    "accepted_extensions",
                    "rejected_extensions",
                    "note",
                    "description",
                }
            }
            for item in sections[section].get("required", [])
        ]

    optional_inputs = sections["inputs"].get("optional", [])
    important_optional_inputs = [
        {
            key: value
            for key, value in item.items()
            if key
            in {
                "name",
                "type",
                "default",
                "minimum",
                "maximum",
                "exclusive_minimum",
                "exclusive_maximum",
                "accepted_forms",
                "rejected_forms",
                "accepted_artifact_semantic_types",
                "rejected_artifact_semantic_types",
                "cardinality",
                "description",
            }
        }
        for item in optional_inputs
        if item.get("name") in {
            "initial_structure",
            "route_parameters",
            "route_section",
            "charge",
            "multiplicity",
            "title",
            "nprocshared",
            "mem",
            "fix_atom_count_line",
            "write_back_fixed_xyz",
            "reference_structure",
        }
    ]
    enumerated_options: list[dict[str, Any]] = []
    for section_name in ("method_spec", "action_settings"):
        section_value = sections[section_name]
        candidates = [
            *section_value.get("required", []),
            *section_value.get("optional_documented", []),
        ]
        for item in candidates:
            if item.get("allowed_values"):
                enumerated_options.append(
                    {
                        "path": f"{section_name}.{item['name']}",
                        "required": bool(item.get("required")),
                        "allowed_values": item["allowed_values"],
                    }
                )

    usage_notes: list[str] = []
    if specification.id == "cluster_conformers" and backend.id == "rdkit":
        usage_notes.append(
            "Pass a typed conformer ensemble, a structured ensemble mapping, or a readable "
            "SDF/MOL/multi-frame XYZ file. A path to a JSON summary is not a conformer file."
        )
    if specification.id == "generate_conformer_ensemble" and backend.id == "crest":
        usage_notes.extend(
            [
                "CREST requires one complete starting geometry with coordinates; do not pass an ensemble or trajectory artifact.",
                "When inputs.initial_structure is supplied it overrides inputs.molecule. Select exactly one frame before submission.",
            ]
        )
    if specification.id == "generate_3d_structure" and backend.id == "rdkit":
        usage_notes.append(
            "inputs.molecule 直接填写 SMILES 时必须使用 RDKit 可解析的标准语法；"
            "不得猜测非标准方括号原子写法。无法确定字面量时，使用上游结构 Action 返回的 ArtifactRef。"
        )
    if (
        specification.id == "generate_gaussian_code"
        and backend.id == "gaussian_codegen"
    ):
        usage_notes.append(
            "inputs.user_input 请求的 Gaussian route 中，solvent 必须使用不含空格的原生别名；"
            "具体拼写必须符合所选 Backend 的原生关键字规则。"
        )
    if (
        specification.id in {"sdf_to_gjf", "xyz_to_gjf"}
        and backend.id == "format_conversion"
    ):
        usage_notes.append(
            "生成的 GaussianInputDeck 在交给 native Gaussian 前由服务统一写入资源指令；"
            "调用方不需要也不能通过格式转换请求选择 CPU 或内存。"
        )
    if specification.id == "sdf_to_gjf" and backend.id == "format_conversion":
        usage_notes.append(
            "sdf_to_gjf 从单记录 SDF 推断电荷和多重度，不接收 inputs.charge 或 "
            "inputs.multiplicity；需要显式填写这两个字段时使用 xyz_to_gjf。"
        )
    if backend.id == "gaussian":
        usage_notes.append(
            "method_spec.method 和 method_spec.basis 必须使用 Gaussian 原生 route 关键字，"
            "并与所选 Backend 的支持范围一致；不得填写其他程序或展示写法。"
        )
        usage_notes.append(
            "method_spec.solvent 必须使用 Gaussian 关键字别名；普通别名不得含空格，"
            "并遵循所选 Backend 的原生拼写要求。"
        )
    handoff_note = _ACTION_INPUT_HANDOFF_NOTES.get(specification.id)
    if handoff_note:
        usage_notes.append(handoff_note)

    compact_contract = {
        "template_kind": "minimal_executable_request",
        "execute_action_request_template": complete[
            "execute_action_request_template"
        ],
        "required_contract": {
            "inputs": required("inputs"),
            "method_spec": required("method_spec"),
            "action_settings": required("action_settings"),
            "component_backends": list(
                backend.required_component_roles.get(specification.id, ())
            ),
        },
        "optional_field_names": {
            "inputs": [item["name"] for item in optional_inputs],
            "method_spec": [
                item["name"]
                for item in sections["method_spec"].get("optional_documented", [])
            ],
            "action_settings": [
                item["name"]
                for item in sections["action_settings"].get(
                    "optional_documented", []
                )
            ],
        },
        "important_optional_inputs": important_optional_inputs,
        "enumerated_options": enumerated_options,
        "conditional_requirements": complete["conditional_requirements"],
        "component_request_contracts": complete["component_request_contracts"],
        "output_contract": complete["output_contract"],
        "input_description": _public_action_text(
            specification.id, "input_description", specification.input_description
        ),
        "description": _public_action_text(
            specification.id, "description", specification.description
        ),
        "usage_notes": usage_notes,
        "retry_policy": (
            "On invalid_request, use the reported diagnostics to correct the same request. "
            "The dispatcher does not choose another provider."
        ),
    }
    if "resource_policy" in complete:
        compact_contract["resource_policy"] = gaussian_resource_policy_record()
    else:
        compact_contract["evaluation_resource_budget"] = dict(_PUBLIC_RESOURCE_BUDGET)
    return compact_contract


def _selection_instruction(specification: ActionSpec) -> str:
    providers = ", ".join(specification.backend_ids)
    if specification.selection_policy == "fixed_source":
        return (
            f"This Data Action uses the fixed source {providers}; omit backend_id and "
            "component_backends. source_id may be omitted or equal the fixed source."
        )
    if specification.selection_policy == "internal_deterministic":
        return (
            f"This deterministic Action uses {providers}; omit backend_id, source_id, and "
            "component_backends."
        )
    if specification.selection_policy == "agent_source_required":
        return f"Select exactly one source_id from: {providers}."
    if specification.selection_policy == "agent_components_required":
        return (
            f"Select exactly one backend_id from: {providers}, then provide every component role "
            "required by that provider."
        )
    return f"Select exactly one backend_id from: {providers}."


def list_action_domains(
    *,
    include_action_ids: bool = True,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _snapshot(snapshot)
    actions = list(current.get("actions", []))
    domains = []
    for category, label in CATEGORY_LABELS.items():
        values = [action for action in actions if action.get("category") == category]
        item = {
                "category": category,
                "description": label,
                "action_count": len(values),
                "scientific_action_count": sum(
                    not bool(action.get("data_action")) for action in values
                ),
                "data_action_count": sum(
                    bool(action.get("data_action")) for action in values
                ),
            }
        if include_action_ids:
            item["action_ids"] = sorted(str(action.get("id")) for action in values)
        domains.append(item)
    return {
        "status": "success",
        **_catalog_identity(current),
        "catalog_visibility_policy": "complete_catalog",
        "task_specific_filtering": False,
        "count": len(domains),
        "includes_action_ids": include_action_ids,
        "domains": domains,
    }


def browse_action_category(
    *,
    category: str,
    action_kind: ActionKind = "all",
    available_only: bool = False,
    detail_level: Literal["summary", "full"] = "full",
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete compact choice set inside one exact Action category."""

    result = search_actions(
        category=category,
        action_kind=action_kind,
        available_only=available_only,
        detail_level=detail_level,
        limit=100,
        snapshot=snapshot,
    )
    result["browse_mode"] = "complete_category"
    result["category_description"] = CATEGORY_LABELS[category]
    return result


def _action_search_fields(
    specifications: list[ActionSpec], backends: Mapping[str, BackendSpec]
) -> dict[str, dict[str, str]]:
    fields: dict[str, dict[str, str]] = {}
    for specification in specifications:
        providers = [backends[item] for item in specification.backend_ids]
        aliases = " ".join(
            dict.fromkeys((*aliases_for_action(specification.id), *specification.aliases))
        )
        fields[specification.id] = {
            "action_id": specification.id.replace("_", " "),
            "aliases": aliases,
            "primary_output": specification.primary_output,
            "category": CATEGORY_LABELS.get(specification.category, ""),
            "description": _public_action_text(
                specification.id, "description", specification.description
            ),
            "keywords": " ".join(specification.keywords),
            "capability_tags": " ".join(specification.capability_tags),
            "scientific_entities": " ".join(specification.scientific_entities),
            "task_verbs": " ".join(specification.task_verbs),
            "input_semantic_types": " ".join(
                specification.input_semantic_types or specification.required_inputs
            ),
            "output_semantic_types": " ".join(
                specification.output_semantic_types or (specification.primary_output,)
            ),
            "input_description": specification.input_description,
            "optional_inputs": " ".join(specification.optional_inputs),
            "backend_ids": " ".join(provider.id for provider in providers),
            "backend_names": " ".join(provider.display_name for provider in providers),
        }
    return fields


def _action_search_documents(fields: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    documents: dict[str, str] = {}
    for action_id, item in fields.items():
        documents[action_id] = weighted_text(
            (
                (item["action_id"], 5),
                (item["aliases"], 4),
                (item["primary_output"], 3),
                (item["category"], 2),
                (item["description"], 2),
                (item["keywords"], 2),
                (item["capability_tags"], 2),
                (item["scientific_entities"], 1),
                (item["task_verbs"], 1),
                (item["input_semantic_types"], 1),
                (item["output_semantic_types"], 1),
                (item["input_description"], 1),
                (item["optional_inputs"], 1),
                (item["backend_ids"], 1),
                (item["backend_names"], 1),
            )
        )
    return documents


def search_actions(
    *,
    query: str | None = None,
    category: str | None = None,
    backend_id: str | None = None,
    action_kind: ActionKind = "all",
    retrieval_mode: Literal["lexical", "hybrid"] = "hybrid",
    detail_level: Literal["summary", "full"] = "full",
    available_only: bool = False,
    limit: int = 20,
    offset: int = 0,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # 所有发现入口都必须复用当前服务绑定的冻结目录；即使不筛选可用性，
    # 也不能重新生成不含健康信息的目录，否则 catalog_hash 会漂移。
    current = _snapshot(snapshot)
    actions = action_specs()
    backends = backend_specs()
    if category is not None and category not in CATEGORY_LABELS:
        raise ValueError(
            f"Unknown category {category!r}; choose one returned by list_action_domains"
        )
    if backend_id is not None and backend_id not in backends:
        raise ValueError(f"Unknown backend_id {backend_id!r}")
    eligible: list[ActionSpec] = []
    for specification in sorted(actions.values(), key=lambda item: item.id):
        if category is not None and specification.category != category:
            continue
        if backend_id is not None and backend_id not in specification.backend_ids:
            continue
        if action_kind == "scientific" and specification.data_action:
            continue
        if action_kind == "data" and not specification.data_action:
            continue
        provider_values = [backends[item] for item in specification.backend_ids]
        if available_only and not any(
            _health_status(current, provider.id) == "available"
            for provider in provider_values
        ):
            continue
        eligible.append(specification)

    search_fields = _action_search_fields(eligible, backends)
    documents = _action_search_documents(search_fields)
    normalized_query = (query or "").casefold().strip()
    semantic_status = "not_requested"
    per_action_scores: dict[str, dict[str, float]] = {}
    per_action_explanations: dict[str, dict[str, Any]] = {}
    predicted_categories: list[dict[str, Any]] = []
    if normalized_query:
        document_vocabulary = sorted(
            {token for document in documents.values() for token in tokenize(document)}
        )
        query_tokens = tokenize(normalized_query)
        query_expansions: dict[str, str] = {}
        for token in query_tokens:
            if token in document_vocabulary or len(token) < 5:
                continue
            matches = difflib.get_close_matches(
                token, document_vocabulary, n=1, cutoff=0.86
            )
            if matches:
                query_expansions[token] = matches[0]
        lexical_query = " ".join(
            [normalized_query, *query_expansions.values()]
        ).strip()
        lexical_ranked = BM25Index(documents).search(lexical_query)
        lexical_raw = {item.document_id: item.score for item in lexical_ranked}
        lexical_terms = {
            item.document_id: list(item.matched_terms) for item in lexical_ranked
        }
        lexical = normalize_scores(lexical_raw)
        semantic_raw: dict[str, float] = {}
        if retrieval_mode == "hybrid":
            semantic_raw, semantic_status = semantic_scores(normalized_query, documents)
        semantic_positive = {
            key: max(0.0, value) for key, value in semantic_raw.items() if value >= 0.20
        }
        semantic = normalize_scores(semantic_positive)
        candidate_ids = set(lexical) | set(semantic)
        for action_id in candidate_ids:
            action_text = action_id.replace("_", " ")
            aliases = aliases_for_action(action_id)
            exact_bonus = 0.0
            if normalized_query in {action_id.casefold(), action_text.casefold()}:
                exact_bonus = 0.35
            elif any(normalized_query == alias.casefold() for alias in aliases):
                exact_bonus = 0.25
            elif normalized_query in documents[action_id].casefold():
                exact_bonus = 0.08
            lexical_score = lexical.get(action_id, 0.0)
            semantic_score = semantic.get(action_id, 0.0)
            combined = lexical_score + 0.12 * semantic_score + exact_bonus
            per_action_scores[action_id] = {
                "combined": combined,
                "lexical": lexical_score,
                "semantic": semantic_score,
                "exact_bonus": exact_bonus,
            }
            query_terms = set(tokenize(normalized_query))
            matched_fields = [
                field_name
                for field_name, field_text in search_fields[action_id].items()
                if query_terms & set(tokenize(field_text))
            ]
            exact_matches = []
            if normalized_query in {action_id.casefold(), action_text.casefold()}:
                exact_matches.append("action_id")
            if any(normalized_query == alias.casefold() for alias in aliases):
                exact_matches.append("alias")
            if exact_matches:
                ranking_reason = "exact action or alias match"
            elif lexical_score and semantic_score:
                ranking_reason = "lexical and semantic relevance"
            elif lexical_score:
                ranking_reason = "lexical relevance"
            else:
                ranking_reason = "semantic relevance"
            per_action_explanations[action_id] = {
                "matched_fields": matched_fields,
                "exact_matches": exact_matches,
                "expanded_terms": sorted(
                    set(lexical_terms.get(action_id, []))
                    | set(query_expansions.values())
                ),
                "ranking_reason": ranking_reason,
            }
        matches = sorted(
            (actions[action_id] for action_id in candidate_ids),
            key=lambda item: (-per_action_scores[item.id]["combined"], item.id),
        )
        ordering = "relevance_with_optional_semantic_support"
        category_scores: dict[str, float] = {}
        for action_id, scores in per_action_scores.items():
            category_name = actions[action_id].category
            category_scores[category_name] = max(
                category_scores.get(category_name, 0.0), scores["combined"]
            )
        predicted_categories = [
            {
                "category": category_name,
                "label": CATEGORY_LABELS[category_name],
                "score": score,
            }
            for category_name, score in sorted(
                category_scores.items(), key=lambda item: (-item[1], item[0])
            )[:3]
        ]
    else:
        matches = eligible
        ordering = "stable_action_id_order"

    page = matches[offset : offset + limit]
    results = []
    for specification in page:
        providers = [
            {
                "backend_id": backend,
                "health_status": _health_status(current, backend),
            }
            for backend in specification.backend_ids
        ]
        summary = {
                "action_id": specification.id,
                "category": specification.category,
        "description": _public_action_text(
            specification.id, "description", specification.description
        ),
                "primary_output": specification.primary_output,
                "data_action": specification.data_action,
                "batch_safe": specification.batch_safe,
                "selection_policy": specification.selection_policy,
                "provider_ids": [item["backend_id"] for item in providers],
                "relevance": per_action_scores.get(specification.id),
                **per_action_explanations.get(
                    specification.id,
                    {
                        "matched_fields": [],
                        "exact_matches": [],
                        "expanded_terms": [],
                        "ranking_reason": "stable catalog order",
                    },
                ),
            }
        if detail_level == "full":
            summary.update(
                {
                    "aliases": list(
                        dict.fromkeys(
                            (*aliases_for_action(specification.id), *specification.aliases)
                        )
                    ),
                    "keywords": list(specification.keywords),
                    "capability_tags": list(specification.capability_tags),
                    "scientific_entities": list(specification.scientific_entities),
                    "task_verbs": list(specification.task_verbs),
                    "input_semantic_types": list(
                        specification.input_semantic_types
                        or specification.required_inputs
                    ),
                    "output_semantic_types": list(
                        specification.output_semantic_types
                        or (specification.primary_output,)
                    ),
                    "providers": providers,
                }
            )
            summary.update(
                {
                    "execution_timeout_policy": _public_timeout_policy(
                        specification.execution_class
                    ),
                    "evaluation_resource_budget": dict(_PUBLIC_RESOURCE_BUDGET),
                }
            )
        results.append(summary)
    category_filter_advisory = None
    if normalized_query and category is not None:
        unrestricted = search_actions(
            query=query,
            category=None,
            backend_id=backend_id,
            action_kind=action_kind,
            retrieval_mode=retrieval_mode,
            detail_level="summary",
            available_only=available_only,
            limit=3,
            offset=0,
            snapshot=current,
        )
        unrestricted_actions = unrestricted.get("actions", [])
        unrestricted_top = unrestricted_actions[0] if unrestricted_actions else None
        filtered_top = results[0] if results else None
        if (
            isinstance(unrestricted_top, dict)
            and unrestricted_top.get("category") != category
            and unrestricted_top.get("exact_matches")
            and not (isinstance(filtered_top, dict) and filtered_top.get("exact_matches"))
        ):
            category_filter_advisory = {
                "code": "exact_match_outside_requested_category",
                "requested_category": category,
                "suggested_category": unrestricted_top.get("category"),
                "suggested_action_id": unrestricted_top.get("action_id"),
                "message": (
                    "The query has a stronger exact action or alias match outside the "
                    "requested category. Inspect the suggested action or retry without "
                    "a category filter."
                ),
            }
        predicted_categories = unrestricted.get(
            "predicted_categories", predicted_categories
        )

    next_offset = offset + len(page)
    return {
        "status": "success",
        **_catalog_identity(current),
        "query": query,
        "filters": {
            "category": category,
            "backend_id": backend_id,
            "action_kind": action_kind,
            "available_only": available_only,
        },
        "ordering": ordering,
        "detail_level": detail_level,
        "predicted_categories": predicted_categories,
        "category_filter_advisory": category_filter_advisory,
        "retrieval": {
            "mode": retrieval_mode,
            "lexical_ranker": "lexical",
            "semantic_model": None,
            "semantic_status": (
                "available"
                if semantic_status in {"ready", "available"}
                else "unavailable"
                if retrieval_mode == "hybrid"
                else "not_requested"
            ),
            "semantic_threshold": None,
            "query_expansions": query_expansions if normalized_query else {},
            "language": "English",
        },
        "total_matches": len(matches),
        "offset": offset,
        "count": len(results),
        "next_offset": next_offset if next_offset < len(matches) else None,
        "actions": results,
    }


def inspect_action(
    action_id: str,
    *,
    backend_id: str | None = None,
    detail_level: Literal["summary", "contract", "full"] = "full",
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _snapshot(snapshot)
    actions = action_specs()
    backends = backend_specs()
    if action_id not in actions:
        raise KeyError(f"Unknown action_id {action_id!r}")
    specification = actions[action_id]
    if backend_id is not None and backend_id not in specification.backend_ids:
        raise ValueError(
            f"Backend {backend_id!r} does not provide {action_id!r}; choose one of "
            f"{list(specification.backend_ids)}"
        )
    selected = (
        [backend_id] if backend_id is not None else list(specification.backend_ids)
    )
    if detail_level == "full":
        provider_contracts = [
            _provider_contract(
                backends[item],
                action_id,
                current,
                detailed=backend_id is not None or len(selected) == 1,
            )
            for item in selected
        ]
    elif backend_id is not None and detail_level == "contract":
        provider_contracts = [
            _provider_summary(backends[backend_id], action_id, current)
        ]
    else:
        provider_contracts = [
            _provider_summary(backends[item], action_id, current) for item in selected
        ]
    selected_request_contract = None
    if backend_id is not None and detail_level == "contract":
        selected_request_contract = _compact_action_request_contract(
            specification, backends[backend_id]
        )
    elif backend_id is not None and detail_level == "full":
        selected_request_contract = _action_request_contract(
            specification, backends[backend_id]
        )
    result = {
        "status": "success",
        **_catalog_identity(current),
        "detail_level": detail_level,
        "action": (
            _public_action_specification(specification)
            if detail_level == "full"
            else _action_summary(specification)
        ),
        "selection_instruction": _selection_instruction(specification),
        "execute_with": "execute_action",
        "action_request_fields": [
            "action_id",
            "backend_id",
            "component_backends",
            "source_id",
            "inputs",
            "method_spec",
            "action_settings",
            *(
                []
                if (
                    backend_id == "gaussian"
                    or specification.primary_output == "GaussianInputDeck"
                )
                else ["resource_limits"]
            ),
        ],
        "provider_contracts": provider_contracts,
        "selected_request_contract": selected_request_contract,
        "detail_note": (
            "The selected provider's compact execution requirements are included."
            if backend_id is not None and detail_level == "contract"
            else "Exact details for the requested provider are included."
            if backend_id is not None or len(selected) == 1
            else "Provider summaries are included. Call inspect_action again with backend_id for "
            "that provider's full parameter, health, runtime, and resource reference."
        ),
        "request_contract_note": (
            "A minimal executable request contract is included for the explicitly selected provider."
            if selected_request_contract is not None and detail_level == "contract"
            else "A complete audit request contract is included for the explicitly selected provider."
            if selected_request_contract is not None
            else "Select one provider and call inspect_action again with backend_id to obtain the "
            "fill-in execute_action request contract."
        ),
        "automatic_fallback": False,
    }
    if specification.batch_safe:
        parallel_execution: dict[str, Any] = {
            "eligible": True,
            "recommended_tool": "submit_action_batch",
            "use_when": (
                "Two or more independent inputs use this same Action and Backend."
            ),
            "warning": (
                "Repeated execute_action calls are synchronous and therefore serial. "
                + (
                    "Gaussian jobs remain service-managed and serial; the caller cannot "
                    "change their CPU or memory."
                    if (
                        backend_id == "gaussian"
                        or specification.primary_output == "GaussianInputDeck"
                    )
                    else "submit_action_batch runs independent children concurrently within the "
                    "active CPU, memory, and GPU budget."
                )
            ),
            "maximum_items": 32,
            "concurrency": (
                (
                    "serial"
                    if (
                        backend_id == "gaussian"
                        or specification.primary_output == "GaussianInputDeck"
                    )
                    else "resource_aware_auto"
                )
            ),
        }
        if backend_id is not None and selected_request_contract is not None:
            execute_template = selected_request_contract[
                "execute_action_request_template"
            ]
            parallel_execution["request_template"] = {
                "action_id": specification.id,
                "backend_id": backend_id,
                "component_backends": execute_template.get(
                    "component_backends", {}
                ),
                "method_spec": execute_template.get("method_spec", {}),
                "action_settings": execute_template.get("action_settings", {}),
                "items": [
                    {
                        "item_id": "<unique item id>",
                        "inputs": execute_template.get("inputs", {}),
                        **(
                            {}
                            if (
                                backend_id == "gaussian"
                                or specification.primary_output == "GaussianInputDeck"
                            )
                            else {
                                "resource_limits": execute_template.get(
                                    "resource_limits", {}
                                )
                            }
                        ),
                    }
                ],
            }
        result["parallel_execution"] = parallel_execution
    service_managed = (
        backend_id == "gaussian"
        or specification.primary_output == "GaussianInputDeck"
    )
    if detail_level in {"contract", "full"} and not service_managed:
        result["execution_timeout_policy"] = _public_timeout_policy(
            specification.execution_class
        )
        result["evaluation_resource_budget"] = dict(_PUBLIC_RESOURCE_BUDGET)
    elif detail_level in {"contract", "full"}:
        result["resource_policy"] = gaussian_resource_policy_record()
    return result


def inspect_backend(
    backend_id: str,
    *,
    action_id: str | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _snapshot(snapshot)
    actions = action_specs()
    backends = backend_specs()
    if backend_id not in backends:
        raise KeyError(f"Unknown backend_id {backend_id!r}")
    backend = backends[backend_id]
    if action_id is not None and action_id not in backend.capabilities:
        raise ValueError(
            f"Backend {backend_id!r} does not provide {action_id!r}; its capabilities are "
            f"{list(backend.capabilities)}"
        )
    capabilities = [
        {
            "action_id": item,
            "category": actions[item].category,
            "description": _public_action_text(
                item,
                "description",
                actions[item].description,
            ),
            "primary_output": actions[item].primary_output,
            "selection_policy": actions[item].selection_policy,
        }
        for item in backend.capabilities
    ]
    result: dict[str, Any] = {
        "status": "success",
        **_catalog_identity(current),
        "backend": {
            "backend_id": backend.id,
            "display_name": backend.display_name,
            "description": _public_backend_text(backend.id, backend.description),
            "runtime": backend.runtime,
            "health": _backend_health(current, backend.id),
            "required_external_data": [],
            "resource_constraints": {
                "service_managed": backend.id == "gaussian",
                "agent_controllable": backend.id != "gaussian",
            },
            "registered_resources": _registered_resources(current, backend.id),
        },
        "capability_count": len(capabilities),
        "capabilities": capabilities,
    }
    if action_id is not None:
        result["action_contract"] = _provider_contract(
            backend, action_id, current, detailed=True
        )
    return result


def search_resources(
    *,
    query: str | None = None,
    backend_id: str | None = None,
    available_only: bool = False,
    limit: int = 20,
    offset: int = 0,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _snapshot(snapshot)
    if backend_id is not None and backend_id not in backend_specs():
        raise ValueError(f"Unknown backend_id {backend_id!r}")
    terms = [term for term in re.split(r"\s+", (query or "").casefold().strip()) if term]
    matches = []
    for resource in sorted(current.get("resources", []), key=lambda item: str(item.get("id"))):
        if backend_id is not None and backend_id not in (
            resource.get("compatible_backends") or []
        ):
            continue
        if available_only and not bool(resource.get("available")):
            continue
        haystack = " ".join(
            str(value)
            for value in (
                resource.get("id"),
                resource.get("display_name"),
                resource.get("kind"),
                resource.get("version"),
                resource.get("format"),
                " ".join(resource.get("compatible_backends") or []),
            )
            if value is not None
        ).casefold()
        if not _matches_all_terms(terms, haystack):
            continue
        matches.append(resource)
    page = matches[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        "status": "success",
        **_catalog_identity(current),
        "ordering": "stable_resource_id_order_not_relevance_ranked",
        "total_matches": len(matches),
        "offset": offset,
        "count": len(page),
        "next_offset": next_offset if next_offset < len(matches) else None,
        "resources": [_resource_summary(resource) for resource in page],
    }


def inspect_resource(
    resource_id: str,
    *,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _snapshot(snapshot)
    for resource in current.get("resources", []):
        if resource.get("id") == resource_id:
            return {
                "status": "success",
                **_catalog_identity(current),
                "resource": _resource_summary(resource),
                "selection_note": (
                    "Use the exact registered selection_syntax and explicitly choose every "
                    "required element, variant, model branch, device, or compatible method."
                ),
            }
    raise KeyError(f"Unknown resource_id {resource_id!r}")


__all__ = [
    "inspect_action",
    "inspect_backend",
    "inspect_resource",
    "list_action_domains",
    "search_actions",
    "search_resources",
]
