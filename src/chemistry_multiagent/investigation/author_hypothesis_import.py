"""作者主假设导入适配器。

本模块只读取仓库内固定夹具，校验题面和结构化计算单元绑定关系，并把冻结的
作者证据写成任务内内容寻址 manifest。它不调用 RetrievalAgent、HypothesisAgent、
Codex 或 MiniChem；Working Memory 的 RetrievalCompleted 和 BatchCreated 仍由既有
服务原子保存，因而崩溃后可以按同一内容和事件键重放而不会生成第二条假设。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from chemistry_multiagent.memory.contracts import (
    ContractValidationError,
    canonical_json,
    validate_structured_hypothesis_claims,
)
from chemistry_multiagent.utils.scientific_input import (
    normalize_calculation_unit,
    normalize_initial_calculation_structures,
    normalize_initial_transition_state_structures,
    normalize_supplementary_scientific_input,
)
from chemistry_multiagent.utils.gaussian_routing import (
    CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE,
)
from chemistry_multiagent.utils.task_artifacts import write_content_addressed_json

from .case3_prepared_inputs import (
    CASE3_PREPARED_INPUT_FILENAMES,
    CASE3_PREPARED_INPUT_SCHEMA,
    build_case3_workspace_input_paths,
    load_case3_prepared_input_bundle as _load_case3_prepared_input_bundle,
    materialize_case3_session_input_files as _materialize_case3_session_input_files,
    validate_case3_prepared_input_metadata as _validate_case3_prepared_input_metadata,
    write_case3_prepared_input_files as _write_case3_prepared_input_files,
)


_SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[3] / "supplementary"
_FIXTURE_PATH = _SUPPLEMENTARY_ROOT / "case1_author_main_hypothesis.json"
_CASE3_FIXTURE_PATH = _SUPPLEMENTARY_ROOT / "case3_author_main_hypothesis.json"
_CASE3_SCIENTIFIC_INPUT_PATH = (
    _SUPPLEMENTARY_ROOT / "case3_author_main_hypothesis_scientific_input.json"
)
_CASE3_PUBLISHED_ARTIFACT_PATH = _SUPPLEMENTARY_ROOT / "case3.json"
_CASE3_HYPOTHESIS_ONLY_FIXTURE_PATH = (
    _SUPPLEMENTARY_ROOT / "case3_author_main_hypothesis_only.json"
)
_CASE3_MAIN_HYPOTHESIS_EXPLORATION_FIXTURE_PATH = (
    _SUPPLEMENTARY_ROOT / "case3_author_main_hypothesis_exploration.json"
)
_TS_COORDINATE_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "supplementary" / (
    "case1_author_ts_scientific_input.json"
)
_MANIFEST_SCHEMA = "arche.investigation.author_hypothesis_import.v1"
_AUTHOR_MANIFEST_DIRECTORY = PurePosixPath(
    "internal/catalog/author_hypothesis_import"
)
_AUTHOR_HYPOTHESIS_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_author_path_like(value: str) -> bool:
    """识别必须留在 CLI 适配层、不能写入任务事实的路径字面量。"""

    normalized = value.replace("\\", "/")
    return (
        value.startswith("/")
        or value.casefold().startswith("file://")
        or value in {"~", "."}
        or value.startswith(("~/", "./"))
        or re.match(r"^[A-Za-z]:/", normalized) is not None
        or any(part == ".." for part in normalized.split("/"))
    )


@dataclass(frozen=True)
class AuthorMainHypothesisDefinition:
    """登记一个作者主假设实验的模式、夹具和公开来源约束。"""

    author_mode: str
    case_profile: str
    fixture_id: str
    fixture_schema: str
    fixture_path: Path
    primary_pdf_sha256: str
    complement_pdf_sha256: str
    primary_pages: list[Any]
    initial_query_angles: tuple[str, ...]
    forbidden_final_value_markers: tuple[str, ...]
    scientific_input_path: Path | None = None
    published_artifact_path: Path | None = None
    published_artifact_sha256: str | None = None
    # advisory 运行配置可以明确复用同一结构夹具的来源 profile；两者不会混用。
    source_fixture_profile: str | None = None


CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE = "author_case1_main_hypothesis"
CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE = "author_case3_main_hypothesis"
CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE = (
    "author_case3_advisory_main_hypothesis"
)
CASE3_AUTHOR_MAIN_HYPOTHESIS_ONLY_MODE = "author_case3_main_hypothesis_only"
CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE = (
    "author_case3_main_hypothesis_exploration"
)
_CASE3_MINIMAL_CONTEXT_MODES = frozenset(
    {
        CASE3_AUTHOR_MAIN_HYPOTHESIS_ONLY_MODE,
        CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE,
    }
)


def build_case3_advisory_constraint_semantics() -> dict[str, Any]:
    """返回 Case 3 advisory 模式的不可变事实与可调整参考约束。

    返回值只描述模型当前可见的四套初始结构边界和作者建议的计算参考，不含作者
    计算结果、能量、排名或结论。调用方每次获得独立副本，便于写入任务事实并在
    恢复请求中逐值复用；模型若调整 advisory reference，必须在结果中说明原因。
    """

    return {
        "schema": "arche.case3.author_main_hypothesis.constraint_semantics.v1",
        "policy": "advisory",
        "immutable_facts": {
            "scientific_object_boundary": (
                "只研究四个固定的 bipyridine-Ni(II)-Br-H 对象："
                "unsubstituted、ortho、meta 和 para；不得增加、删除或替换对象。"
            ),
            "initial_calculation_structures": {
                "identities": ["unsubstituted", "ortho", "meta", "para"],
                "fields": [
                    "atomic_numbers",
                    "coordinates_angstrom",
                    "charge",
                    "multiplicity",
                    "coordinate_atom_indices",
                    "tool_coordinate_atom_indices",
                    "atom_roles",
                ],
                "coordinate_roles": ["Br", "N", "N", "H"],
                "coordinate_index_convention": (
                    "coordinate_atom_indices 是输入记录的 1-based 索引；"
                    "tool_coordinate_atom_indices 是对应的 MiniChem 0-based 索引。"
                ),
            },
        },
        "advisory_references": {
            "method": "wB97XD",
            "basis": "def2SVP",
            "solvent_model": "SMD(DMA)",
            "solvent": "N,N-Dimethylacetamide",
            "symmetry": "NoSymm",
            "relaxed_scan": {
                "coordinate_type": "dihedral",
                "nominal_start_degrees": 0,
                "nominal_end_degrees": 60,
                "nominal_step_degrees": 5,
                "nominal_relaxed": True,
                "scan_origin": "input_geometry",
                "nominal_positive_displacement_degrees": 60,
                "absolute_tool_endpoints_degrees": {
                    "meta": {
                        "start": -0.020536359293,
                        "end": 59.979463640707,
                    },
                    "ortho": {
                        "start": -4.040493659638,
                        "end": 55.959506340362,
                    },
                    "para": {
                        "start": -0.007200953529,
                        "end": 59.992799046471,
                    },
                    "unsubstituted": {
                        "start": 0.003210601878,
                        "end": 60.003210601878,
                    },
                },
            },
        },
        "model_obligation": (
            "immutable facts 和科学对象边界不得改变；advisory references 可以采用或调整。"
            "在工具支持且没有本次任务的失败证据时，优先尝试作者基线；其中 DMA 的准确"
            "化学名称是 N,N-Dimethylacetamide，不得将缩写自行展开为另一种溶剂。这个顺序"
            "建议不是硬约束。作者扫描基线从每套输入几何出发正向位移 60°；需要绝对端点"
            "的工具使用 absolute_tool_endpoints_degrees。遇到工具失败或科学证据时仍可调整。"
            "采用或调整这些参考时，"
            "都必须说明选择及其对可比性的影响；所有能量、曲线、排序和科学结论只能来自"
            "本次实际工具结果。"
        ),
    }

AUTHOR_MAIN_HYPOTHESIS_DEFINITIONS: dict[str, AuthorMainHypothesisDefinition] = {
    CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE: AuthorMainHypothesisDefinition(
        author_mode=CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE,
        case_profile="case1_fixed_reproduction",
        fixture_id="case1_author_main_hypothesis",
        fixture_schema="arche.case1.author_main_hypothesis",
        fixture_path=_FIXTURE_PATH,
        primary_pdf_sha256=(
            "98b98fc1c0cde1a07825508c7682dbe6fca07748f3376a48c2f1ce5862a2cdff"
        ),
        complement_pdf_sha256=(
            "5e602e3a3a96c0f23239a35244f6faf51d4b48aaecc05660956612a0657112fb"
        ),
        primary_pages=[20, "24-25"],
        initial_query_angles=("关键 C–C 成键过渡态的对映路径比较",),
        forbidden_final_value_markers=("TS-2a", "TS-2b", "1.88", "88%"),
    ),
    CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE: AuthorMainHypothesisDefinition(
        author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE,
        case_profile="case3_fixed_reproduction",
        fixture_id="case3_author_main_hypothesis",
        fixture_schema="arche.case3.author_main_hypothesis",
        fixture_path=_CASE3_FIXTURE_PATH,
        primary_pdf_sha256=(
            "1bd895242da54eee460e0220d728e4ce97b8fa9bb6c71cc75b4531a1b65dd496"
        ),
        complement_pdf_sha256=(
            "b2f5aacac2b58b62baaa8b43edff451ddffd2520c319da6c39031e2e86eed70f"
        ),
        primary_pages=[12, 13],
        initial_query_angles=(
            "四种 bipyridine 配体的 Br–N–N–H relaxed dihedral 能量曲线比较",
        ),
        # 只阻断作者发布的数值答案；模型仍可看到题面中的实验选择性描述。
        forbidden_final_value_markers=("12.8", "22.0", "12．8", "22．0"),
        scientific_input_path=_CASE3_SCIENTIFIC_INPUT_PATH,
        published_artifact_path=_CASE3_PUBLISHED_ARTIFACT_PATH,
        published_artifact_sha256=(
            "718959095618dd644effbe09478d9dcf3ea3d7225320159015ef24e6ec21b95e"
        ),
    ),
    CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE: AuthorMainHypothesisDefinition(
        author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE,
        case_profile=CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE,
        fixture_id="case3_author_main_hypothesis",
        fixture_schema="arche.case3.author_main_hypothesis",
        fixture_path=_CASE3_FIXTURE_PATH,
        primary_pdf_sha256=(
            "1bd895242da54eee460e0220d728e4ce97b8fa9bb6c71cc75b4531a1b65dd496"
        ),
        complement_pdf_sha256=(
            "b2f5aacac2b58b62baaa8b43edff451ddffd2520c319da6c39031e2e86eed70f"
        ),
        primary_pages=[12, 13],
        initial_query_angles=(
            "四种 bipyridine 配体的 Br–N–N–H relaxed dihedral 能量曲线比较",
        ),
        forbidden_final_value_markers=("12.8", "22.0", "12．8", "22．0"),
        scientific_input_path=_CASE3_SCIENTIFIC_INPUT_PATH,
        published_artifact_path=_CASE3_PUBLISHED_ARTIFACT_PATH,
        published_artifact_sha256=(
            "718959095618dd644effbe09478d9dcf3ea3d7225320159015ef24e6ec21b95e"
        ),
        source_fixture_profile="case3_fixed_reproduction",
    ),
    CASE3_AUTHOR_MAIN_HYPOTHESIS_ONLY_MODE: AuthorMainHypothesisDefinition(
        author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_ONLY_MODE,
        case_profile="case3_fixed_reproduction",
        fixture_id="case3_author_main_hypothesis_only",
        fixture_schema="arche.case3.author_main_hypothesis_only",
        fixture_path=_CASE3_HYPOTHESIS_ONLY_FIXTURE_PATH,
        primary_pdf_sha256=(
            "1bd895242da54eee460e0220d728e4ce97b8fa9bb6c71cc75b4531a1b65dd496"
        ),
        complement_pdf_sha256=(
            "b2f5aacac2b58b62baaa8b43edff451ddffd2520c319da6c39031e2e86eed70f"
        ),
        primary_pages=[12, 13],
        initial_query_angles=(),
        forbidden_final_value_markers=("12.8", "22.0", "12．8", "22．0"),
    ),
    CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE: AuthorMainHypothesisDefinition(
        author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE,
        case_profile="case3_fixed_reproduction",
        fixture_id="case3_author_main_hypothesis_exploration",
        fixture_schema="arche.case3.author_main_hypothesis_exploration",
        fixture_path=_CASE3_MAIN_HYPOTHESIS_EXPLORATION_FIXTURE_PATH,
        primary_pdf_sha256=(
            "1bd895242da54eee460e0220d728e4ce97b8fa9bb6c71cc75b4531a1b65dd496"
        ),
        complement_pdf_sha256=(
            "b2f5aacac2b58b62baaa8b43edff451ddffd2520c319da6c39031e2e86eed70f"
        ),
        primary_pages=[12, 13],
        initial_query_angles=(),
        forbidden_final_value_markers=("12.8", "22.0", "12．8", "22．0"),
    ),
}


def author_main_hypothesis_definition(author_mode: str) -> AuthorMainHypothesisDefinition:
    """返回已登记的作者模式定义，未知模式立即拒绝。"""

    definition = AUTHOR_MAIN_HYPOTHESIS_DEFINITIONS.get(author_mode)
    if definition is None:
        raise ContractValidationError("作者主假设模式未注册")
    return definition


def author_main_hypothesis_mode_for_case_profile(case_profile: str) -> str:
    """根据固定 Case profile 返回对应作者模式。"""

    matches = [
        definition.author_mode
        for definition in AUTHOR_MAIN_HYPOTHESIS_DEFINITIONS.values()
        if definition.case_profile == case_profile
    ]
    if len(matches) > 1 and case_profile == "case3_fixed_reproduction":
        return CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE
    if len(matches) != 1:
        raise ContractValidationError("Case profile 没有唯一作者主假设模式")
    return matches[0]


def is_author_main_hypothesis_mode(value: Any) -> bool:
    """判断一个值是否为已登记的作者单假设模式。"""

    return isinstance(value, str) and value in AUTHOR_MAIN_HYPOTHESIS_DEFINITIONS


def _validate_author_stable_reference(value: Any, field_name: str) -> str:
    """校验作者来源的稳定引用，拒绝把调用方路径写进任务事实。"""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractValidationError(f"{field_name} 必须是无首尾空白的非空引用")
    if "\x00" in value or "\\" in value:
        raise ContractValidationError(f"{field_name} 不能包含本机路径或控制字符")
    if _is_author_path_like(value):
        raise ContractValidationError(f"{field_name} 不能是调用方路径")
    return value


def _reject_author_path_values(value: Any, field_name: str) -> None:
    """递归检查 provenance/hypothesis，阻止隐含的调用方文件路径。"""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ContractValidationError(f"{field_name} 的字段名必须是非空文本")
            _reject_author_path_values(item, f"{field_name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_author_path_values(item, f"{field_name}[{index}]")
        return
    if isinstance(value, str) and _is_author_path_like(value):
        raise ContractValidationError(f"{field_name} 不能包含调用方路径")


def normalize_author_hypotheses(value: Any) -> list[dict[str, Any]]:
    """校验并复制通用作者假设直入的 TaskCreated 输入。

    输入是研究员提供的 1–5 条结构化作者假设；动作只校验稳定来源引用、
    SHA-256、完整科学正文和 provenance，不读取文件、不调用模型或服务，也不
    接受 ``materials``。返回值是规范 JSON 副本；未知字段、重复身份、调用方路径
    或旧版假设正文会抛出 ``ContractValidationError``。
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ContractValidationError("author_hypotheses 必须是 1–5 条记录的数组")
    if not 1 <= len(value) <= 5:
        raise ContractValidationError("author_hypotheses 必须包含 1–5 条记录")

    normalized: list[dict[str, Any]] = []
    hypothesis_refs: set[str] = set()
    required_fields = {
        "hypothesis_ref",
        "source_ref",
        "source_digest",
        "hypothesis",
        "provenance",
    }
    for index, raw_item in enumerate(value):
        field_name = f"author_hypotheses[{index}]"
        if not isinstance(raw_item, Mapping):
            raise ContractValidationError(f"{field_name} 必须是 JSON object")
        if set(raw_item) != required_fields:
            raise ContractValidationError(
                f"{field_name} 字段必须恰好为 {', '.join(sorted(required_fields))}"
            )
        hypothesis_ref = _validate_author_stable_reference(
            raw_item["hypothesis_ref"], f"{field_name}.hypothesis_ref"
        )
        if hypothesis_ref in hypothesis_refs:
            raise ContractValidationError(
                "author_hypotheses 不能重复 hypothesis_ref"
            )
        hypothesis_refs.add(hypothesis_ref)
        source_ref = _validate_author_stable_reference(
            raw_item["source_ref"], f"{field_name}.source_ref"
        )
        source_digest = raw_item["source_digest"]
        if not isinstance(source_digest, str) or not _AUTHOR_HYPOTHESIS_SHA256_RE.fullmatch(
            source_digest
        ):
            raise ContractValidationError(
                f"{field_name}.source_digest 必须是 64 位小写 SHA-256"
            )
        hypothesis = raw_item["hypothesis"]
        if not isinstance(hypothesis, Mapping) or not hypothesis:
            raise ContractValidationError(
                f"{field_name}.hypothesis 必须是完整 JSON object"
            )
        try:
            _reject_author_path_values(hypothesis, f"{field_name}.hypothesis")
            validate_structured_hypothesis_claims(
                hypothesis,
                path=f"{field_name}.hypothesis",
                allow_legacy=False,
            )
            hypothesis_copy = json.loads(canonical_json(dict(hypothesis)))
        except (ContractValidationError, TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"{field_name}.hypothesis 不满足结构化作者假设合同"
            ) from exc
        _reject_author_path_values(hypothesis_copy, f"{field_name}.hypothesis")
        embedded_ref = hypothesis_copy.get(
            "hypothesis_id", hypothesis_copy.get("hypothesis_ref")
        )
        if embedded_ref is not None and embedded_ref != hypothesis_ref:
            raise ContractValidationError(
                f"{field_name}.hypothesis 内部身份与 hypothesis_ref 不一致"
            )
        provenance = raw_item["provenance"]
        if not isinstance(provenance, Mapping) or not provenance:
            raise ContractValidationError(
                f"{field_name}.provenance 必须是完整 JSON object"
            )
        _reject_author_path_values(provenance, f"{field_name}.provenance")
        try:
            provenance_copy = json.loads(canonical_json(dict(provenance)))
        except (ContractValidationError, TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"{field_name}.provenance 不是严格 JSON object"
            ) from exc
        normalized.append(
            {
                "hypothesis_ref": hypothesis_ref,
                "source_ref": source_ref,
                "source_digest": source_digest,
                "hypothesis": hypothesis_copy,
                "provenance": provenance_copy,
            }
        )
    return normalized


@dataclass(frozen=True)
class AuthorMainHypothesisFixture:
    """经过完整校验的作者夹具及其内容摘要。"""

    fixture_id: str
    fixture_sha256: str
    case_profile: str
    question_sha256: str
    calculation_unit_sha256: str
    calculation_unit: Mapping[str, Any]
    initial_transition_state_structures: Mapping[str, Any] | None
    initial_calculation_structures: Mapping[str, Any] | None
    retrieval_context: Mapping[str, Any]
    hypothesis: Mapping[str, Any]
    selection: Mapping[str, Any]
    provenance: Mapping[str, Any]
    author_mode: str
    initial_query_angles: tuple[str, ...]
    constraint_semantics: Mapping[str, Any] | None = None


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    """对 JSON object 做规范编码并返回 SHA-256，不写入外部状态。"""

    return hashlib.sha256(canonical_json(dict(value)).encode("utf-8")).hexdigest()


def _content_addressed_ref(value: Mapping[str, Any], stem: str) -> str:
    """按 ``write_content_addressed_json`` 的规则计算任务内相对引用。"""

    digest = _canonical_sha256(value)
    return (_AUTHOR_MANIFEST_DIRECTORY / f"{stem}_{digest}.json").as_posix()


def _read_content_addressed_json(
    task_root: str | Path,
    relative_ref: Any,
    label: str,
) -> dict[str, Any]:
    """读取并核对作者导入 artifact 的安全路径、文件名摘要和规范字节。"""

    if not isinstance(relative_ref, str) or not relative_ref or "\\" in relative_ref:
        raise ContractValidationError(f"{label} 必须是安全相对路径")
    relative = PurePosixPath(relative_ref)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ContractValidationError(f"{label} 必须是安全相对路径")
    root = Path(task_root).resolve(strict=False)
    target = root.joinpath(*relative.parts)
    try:
        target.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractValidationError(f"{label} 越出任务目录或文件不可用") from exc
    if target.is_symlink() or not target.is_file():
        raise ContractValidationError(f"{label} 不存在或不是普通文件")
    try:
        raw = target.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"{label} 无法严格解析") from exc
    if not isinstance(parsed, dict):
        raise ContractValidationError(f"{label} 必须是 JSON object")
    digest = hashlib.sha256(raw).hexdigest()
    if target.name != f"{target.name.split('_', 1)[0]}_{digest}.json":
        raise ContractValidationError(f"{label} 文件名摘要不一致")
    try:
        canonical_bytes = canonical_json(parsed).encode("utf-8")
    except ContractValidationError as exc:
        raise ContractValidationError(f"{label} 不是严格 JSON") from exc
    if raw != canonical_bytes:
        raise ContractValidationError(f"{label} 内容不是规范 JSON")
    return parsed


def validate_case3_prepared_input_metadata(value: Any) -> dict[str, Any]:
    """校验作者导入所需的 prepared metadata。

    输入是 TaskCreated 中的冻结 JSON object；动作调用专用五文件合同做内存校验，
    不读取外部目录、raw GJF 或日志。返回规范普通字典；字段或摘要非法时以
    ContractValidationError 失败。
    """

    try:
        return _validate_case3_prepared_input_metadata(value)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc


def load_case3_prepared_input_bundle(directory: str | Path) -> dict[str, Any]:
    """读取父流程 prepared bundle 并转换为统一合同异常。

    输入是待导入的绝对目录；动作只读取五个清理文件并生成冻结内容，不解析 GJF，
    不读取 raw 来源或日志，也不修改外部目录。返回可写入 TaskCreated 的普通字典；
    目录或文件不符合合同则抛出 ContractValidationError。
    """

    try:
        return _load_case3_prepared_input_bundle(directory)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc


def write_case3_prepared_input_files(
    task_root: str | Path,
    structure_input: Mapping[str, Any],
) -> dict[str, Any]:
    """把冻结清理内容写入任务目录并转换为统一合同异常。

    输入是已冻结的任务根和 prepared metadata；动作只在任务内原子复制五个文件，
    不读取外部来源。返回规范 metadata；目录、摘要或文件类型非法时抛出
    ContractValidationError。
    """

    try:
        return _write_case3_prepared_input_files(task_root, structure_input)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc


def materialize_case3_session_input_files(
    task_root: str | Path,
    workspace: str | Path,
    structure_input: Mapping[str, Any],
) -> None:
    """把任务内冻结清理内容复制到 session workspace。

    输入是任务根、session workspace 和冻结 metadata；动作只从任务内内容核对摘要并
    原子复制五个文件，不读取外部作者目录、raw GJF 或日志。成功无返回值；路径、摘要
    或文件类型非法时抛出 ContractValidationError。
    """

    try:
        _materialize_case3_session_input_files(task_root, workspace, structure_input)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc


def _read_fixture_json(
    definition: AuthorMainHypothesisDefinition,
) -> dict[str, Any]:
    """读取登记夹具并拒绝符号链接、重复键和非法 JSON 常量。"""

    # Case 1 旧测试与本地探针会替换模块级路径；Case3 路径必须由登记定义固定。
    fixture_path = (
        _FIXTURE_PATH
        if definition.author_mode == CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE
        else definition.fixture_path
    )
    if fixture_path.is_symlink() or not fixture_path.is_file():
        raise ContractValidationError(
            f"{definition.case_profile} 作者主假设夹具缺失或不是普通文件"
        )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ContractValidationError("作者主假设夹具含重复字段")
            result[key] = item
        return result

    try:
        return json.loads(
            fixture_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractValidationError(f"作者主假设夹具含非法数字：{value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("作者主假设夹具无法严格解析") from exc


def _read_author_ts_coordinate_fixture() -> dict[str, Any]:
    """读取并校验版本控制的两套作者 TS 初始坐标输入。"""

    if (
        _TS_COORDINATE_FIXTURE_PATH.is_symlink()
        or not _TS_COORDINATE_FIXTURE_PATH.is_file()
    ):
        raise ContractValidationError("Case 1 作者 TS 坐标夹具缺失或不是普通文件")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ContractValidationError("作者 TS 坐标夹具含重复字段")
            result[key] = item
        return result

    try:
        raw = json.loads(
            _TS_COORDINATE_FIXTURE_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractValidationError(f"作者 TS 坐标夹具含非法数字：{value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("作者 TS 坐标夹具无法严格解析") from exc
    if not isinstance(raw, Mapping) or set(raw) != {
        "calculation_unit",
        "initial_transition_state_structures",
    }:
        raise ContractValidationError("作者 TS 坐标夹具字段不完整或含未知字段")
    try:
        normalized = normalize_initial_transition_state_structures(
            raw["initial_transition_state_structures"]
        )
    except ValueError as exc:
        raise ContractValidationError("作者 TS 坐标夹具合同非法") from exc
    if set(normalized) != {"TS-2a", "TS-2b-1"}:
        raise ContractValidationError(
            "作者 TS 坐标夹具必须恰好包含 TS-2a 和 TS-2b-1"
        )
    atomnos_by_identity = {
        identity: record["atomic_numbers"] for identity, record in normalized.items()
    }
    if any(len(atomnos) != 53 for atomnos in atomnos_by_identity.values()):
        raise ContractValidationError("作者 TS 坐标夹具必须固定为 53 个原子")
    if atomnos_by_identity["TS-2a"] != atomnos_by_identity["TS-2b-1"]:
        raise ContractValidationError("作者 TS 坐标夹具两套结构必须使用相同原子序数")
    for identity, record in normalized.items():
        if record["charge"] != 1 or record["multiplicity"] != 1:
            raise ContractValidationError(
                f"作者 TS 坐标夹具 {identity} 必须固定为 charge=1、multiplicity=1"
            )
    return normalized


def _read_case3_published_initial_structure_sources(
    definition: AuthorMainHypothesisDefinition,
) -> dict[str, dict[str, Any]]:
    """从作者发布 artifact 提取四套首点事实，仅返回结构与坐标索引。

    输入来自模式定义登记的受版本控制 JSON；函数先核对文件身份和 SHA-256，再
    读取 ``dihedral.scan`` 的首个坐标、原子序数、电子态和唯一 Gaussian
    ``D(...)`` 索引。能量、后续扫描点和结果摘要不会进入返回值。文件漂移、结构
    缺失或扫描名不唯一时抛出合同错误。
    """

    artifact_path = definition.published_artifact_path
    expected_sha256 = definition.published_artifact_sha256
    if artifact_path is None or expected_sha256 is None:
        raise ContractValidationError("Case 3 作者发布 artifact 未登记")
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ContractValidationError("Case 3 作者发布 artifact 缺失或不是普通文件")
    try:
        raw_bytes = artifact_path.read_bytes()
    except OSError as exc:
        raise ContractValidationError("Case 3 作者发布 artifact 无法读取") from exc
    if hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
        raise ContractValidationError("Case 3 作者发布 artifact SHA-256 漂移")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ContractValidationError("Case 3 作者发布 artifact 含重复字段")
            result[key] = item
        return result

    try:
        raw = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractValidationError(
                    f"Case 3 作者发布 artifact 含非法数字：{value}"
                )
            ),
        )
        scans = raw["dihedral"]["scan"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContractValidationError(
            "Case 3 作者发布 artifact 缺少 dihedral.scan"
        ) from exc
    expected_identities = {"unsubstituted", "ortho", "meta", "para"}
    if not isinstance(scans, Mapping) or set(scans) != expected_identities:
        raise ContractValidationError("Case 3 作者发布 artifact 的扫描身份不完整")

    extracted: dict[str, dict[str, Any]] = {}
    for identity, scan_record in scans.items():
        if not isinstance(scan_record, Mapping):
            raise ContractValidationError(
                f"Case 3 作者发布 artifact 的 {identity} 扫描不是 object"
            )
        scan_steps = scan_record.get("scan_steps")
        scan_summary = scan_record.get("scan_summary")
        if (
            not isinstance(scan_steps, list)
            or not scan_steps
            or not isinstance(scan_steps[0], Mapping)
            or not isinstance(scan_summary, Mapping)
        ):
            raise ContractValidationError(
                f"Case 3 作者发布 artifact 的 {identity} 缺少首扫描点"
            )
        scan_names = scan_summary.get("scan_names")
        if not isinstance(scan_names, list) or not scan_names:
            raise ContractValidationError(
                f"Case 3 作者发布 artifact 的 {identity} 缺少扫描名"
            )
        unique_scan_names = {
            item.strip() for item in scan_names if isinstance(item, str) and item.strip()
        }
        if len(unique_scan_names) != 1:
            raise ContractValidationError(
                f"Case 3 作者发布 artifact 的 {identity} 没有唯一二面角"
            )
        match = re.fullmatch(
            r"D\((\d+),(\d+),(\d+),(\d+)\)",
            next(iter(unique_scan_names)),
        )
        if match is None:
            raise ContractValidationError(
                f"Case 3 作者发布 artifact 的 {identity} 二面角格式非法"
            )
        extracted[identity] = {
            "atomic_numbers": scan_record.get("atomnos"),
            "coordinates_angstrom": scan_steps[0].get("atomcoords"),
            "charge": scan_record.get("charge"),
            "multiplicity": scan_record.get("mult"),
            "coordinate_atom_indices": [int(item) for item in match.groups()],
        }
    return extracted


def _read_case3_scientific_input_fixture(
    definition: AuthorMainHypothesisDefinition,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取 Case 3 作者发布首点结构夹具并拒绝计算结果字段。

    该夹具只保存四种配体的第一扫描点、电子态和共同 relaxed scan 协议；它是
    published initial geometry reproduction，不是题面自主生成结构。函数不读取
    ``supplementary/case3.json`` 运行结果，也不接受能量、后续点或排序字段。
    """

    fixture_path = definition.scientific_input_path
    if fixture_path is None:
        raise ContractValidationError("Case 3 作者首点结构夹具未登记")
    if fixture_path.is_symlink() or not fixture_path.is_file():
        raise ContractValidationError("Case 3 作者首点结构夹具缺失或不是普通文件")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ContractValidationError("Case 3 作者首点结构夹具含重复字段")
            result[key] = item
        return result

    try:
        raw = json.loads(
            fixture_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractValidationError(f"Case 3 作者首点结构夹具含非法数字：{value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("Case 3 作者首点结构夹具无法严格解析") from exc
    if not isinstance(raw, Mapping) or set(raw) != {
        "calculation_unit",
        "initial_calculation_structures",
    }:
        raise ContractValidationError("Case 3 作者首点结构夹具字段不完整或含未知字段")
    try:
        calculation_unit = normalize_calculation_unit(raw["calculation_unit"])
        structures = normalize_initial_calculation_structures(
            raw["initial_calculation_structures"]
        )
    except ValueError as exc:
        raise ContractValidationError("Case 3 作者首点结构夹具合同非法") from exc
    if set(structures) != {"unsubstituted", "ortho", "meta", "para"}:
        raise ContractValidationError(
            "Case 3 作者首点结构夹具必须恰好包含四种配体"
        )
    expected_atom_roles = ["Br", "N", "N", "H"]
    expected_scan = {
        "coordinate_type": "dihedral",
        "start_degrees": 0,
        "end_degrees": 60,
        "step_degrees": 5,
        "point_count": 13,
        "tool_step_count": 12,
        "tool_value_unit": "degree",
        "relaxed": True,
        "method": "wB97XD",
        "basis": "def2-SVP",
        "solvent_model": "SMD(DMA)",
        "symmetry": "NoSymm",
    }
    published_sources = _read_case3_published_initial_structure_sources(definition)
    for identity, record in structures.items():
        if record["charge"] != 0 or record["multiplicity"] != 1:
            raise ContractValidationError(
                f"Case 3 作者首点结构 {identity} 必须固定为 charge=0、multiplicity=1"
            )
        if record["atom_roles"] != expected_atom_roles:
            raise ContractValidationError(
                f"Case 3 作者首点结构 {identity} 的原子角色必须是 Br–N–N–H"
            )
        if record["scan"] != expected_scan:
            raise ContractValidationError(
                f"Case 3 作者首点结构 {identity} 的 scan 协议不匹配 published route"
            )
        source = published_sources[identity]
        for field in (
            "atomic_numbers",
            "coordinates_angstrom",
            "charge",
            "multiplicity",
            "coordinate_atom_indices",
        ):
            if record[field] != source[field]:
                raise ContractValidationError(
                    f"Case 3 作者首点结构 {identity} 的 {field} 与发布 artifact 不一致"
                )
        expected_atomic_numbers = [35, 7, 7, 1]
        coordinate_atomic_numbers = [
            record["atomic_numbers"][index - 1]
            for index in record["coordinate_atom_indices"]
        ]
        if coordinate_atomic_numbers != expected_atomic_numbers:
            raise ContractValidationError(
                f"Case 3 作者首点结构 {identity} 的坐标元素不是 Br–N–N–H"
            )
        if any(
            forbidden in record
            for forbidden in (
                "energy",
                "energy_ev",
                "scan_steps",
                "scan_summary",
                "ranking",
                "result",
            )
        ):
            raise ContractValidationError(
                f"Case 3 作者首点结构 {identity} 携带禁止的结果字段"
            )
    return calculation_unit, structures


def _reject_final_answer_markers(
    value: Any,
    markers: tuple[str, ...],
) -> None:
    """递归拒绝作者最终数值或标签进入冻结夹具正文。"""

    if isinstance(value, Mapping):
        for item in value.values():
            _reject_final_answer_markers(item, markers)
    elif isinstance(value, list):
        for item in value:
            _reject_final_answer_markers(item, markers)
    elif isinstance(value, str) and any(marker in value for marker in markers):
        raise ContractValidationError(
            "作者主假设夹具不得携带作者最终数值答案或结果标签"
        )


def _require_object(value: Any, label: str) -> dict[str, Any]:
    """确认夹具分区是 JSON object，并返回独立副本。"""

    if not isinstance(value, dict):
        raise ContractValidationError(f"{label} 必须是 JSON object")
    return copy.deepcopy(value)


def load_author_main_hypothesis(
    scientific_question: str,
    *,
    author_mode: str,
    supplementary_scientific_input: Mapping[str, Any] | None = None,
) -> AuthorMainHypothesisFixture:
    """校验一个登记作者夹具与当前题面/计算单元的绑定关系。

    输入是当前公开题面、作者模式和可选结构化计算单元；动作只读取对应的仓库
    夹具并核对 provenance、哈希、科学字段和答案隔离，不修改文件或调用模型。
    返回冻结夹具对象；题面、上下文、来源哈希或夹具正文漂移时抛出合同错误。
    """

    if not isinstance(scientific_question, str) or not scientific_question.strip():
        raise ContractValidationError("scientific_question 必须是非空字符串")
    definition = author_main_hypothesis_definition(author_mode)
    minimal_context_mode = author_mode in _CASE3_MINIMAL_CONTEXT_MODES
    raw = _read_fixture_json(definition)
    _reject_final_answer_markers(raw, definition.forbidden_final_value_markers)
    required = (
        {
            "schema",
            "schema_version",
            "fixture_id",
            "case_profile",
            "question_binding",
            "hypothesis",
            "provenance",
        }
        if minimal_context_mode
        else {
            "schema",
            "schema_version",
            "fixture_id",
            "case_profile",
            "question_binding",
            "context_binding",
            "calculation_unit",
            "retrieval_context",
            "hypothesis",
            "selection",
            "provenance",
        }
    )
    if set(raw) != required:
        raise ContractValidationError("作者主假设夹具字段不完整或含未知字段")
    if (
        raw["schema"] != definition.fixture_schema
        or raw["schema_version"] != 1
        or raw["fixture_id"] != definition.fixture_id
        or raw["case_profile"]
        != (definition.source_fixture_profile or definition.case_profile)
    ):
        raise ContractValidationError("作者主假设夹具 schema 或身份非法")
    question_binding = _require_object(raw["question_binding"], "question_binding")
    if set(question_binding) != {"method", "question_sha256"}:
        raise ContractValidationError("question_binding 字段非法")
    question_sha256 = hashlib.sha256(scientific_question.encode("utf-8")).hexdigest()
    if (
        question_binding["method"] != "sha256_utf8_exact_question"
        or question_binding["question_sha256"] != question_sha256
    ):
        raise ContractValidationError("作者主假设夹具与当前题面不匹配")
    initial_transition_state_structures = None
    initial_calculation_structures = None
    if minimal_context_mode:
        calculation_unit = {}
        calculation_unit_sha256 = _canonical_sha256(calculation_unit)
        if supplementary_scientific_input is not None:
            try:
                provided = normalize_supplementary_scientific_input(
                    supplementary_scientific_input
                )
            except ValueError as exc:
                raise ContractValidationError(
                    "Case 3 精简主假设模式不接受结构化科学输入"
                ) from exc
            if provided:
                raise ContractValidationError(
                    "Case 3 精简主假设模式不接受结构化科学输入"
                )
        retrieval_context = {}
    else:
        context_binding = _require_object(raw["context_binding"], "context_binding")
        expected_context_binding_fields = {
            "method",
            "calculation_unit_sha256",
        }
        if definition.scientific_input_path is not None:
            expected_context_binding_fields.add("initial_calculation_structures_sha256")
        if set(context_binding) != expected_context_binding_fields:
            raise ContractValidationError("作者主假设输入字段非法")
        try:
            calculation_unit = normalize_calculation_unit(raw["calculation_unit"])
        except ValueError as exc:
            raise ContractValidationError("作者主假设输入 calculation_unit 非法") from exc
        calculation_unit_sha256 = _canonical_sha256(calculation_unit)
        expected_context_method = (
            "sha256_canonical_normalized_calculation_unit_and_initial_calculation_structures"
            if definition.scientific_input_path is not None
            else "sha256_canonical_normalized_calculation_unit"
        )
        if (
            context_binding["method"] != expected_context_method
            or context_binding["calculation_unit_sha256"] != calculation_unit_sha256
        ):
            raise ContractValidationError("作者主假设输入与计算单元绑定摘要不匹配")
        if definition.scientific_input_path is not None:
            fixture_calculation_unit, fixture_structures = (
                _read_case3_scientific_input_fixture(definition)
            )
            if fixture_calculation_unit != calculation_unit:
                raise ContractValidationError(
                    "Case 3 作者首点结构输入与作者 calculation_unit 不一致"
                )
            structure_sha256 = _canonical_sha256(fixture_structures)
            if context_binding["initial_calculation_structures_sha256"] != structure_sha256:
                raise ContractValidationError(
                    "Case 3 作者首点结构输入与作者 fixture 绑定摘要不匹配"
                )
            initial_calculation_structures = copy.deepcopy(fixture_structures)
        if supplementary_scientific_input is not None:
            try:
                provided = normalize_supplementary_scientific_input(
                    supplementary_scientific_input
                )
            except ValueError as exc:
                raise ContractValidationError("作者模式结构化计算单元非法") from exc
            if provided.get("calculation_unit") != calculation_unit:
                raise ContractValidationError("作者模式拒绝漂移的结构化计算单元")
            if "initial_transition_state_structures" in provided:
                expected_coordinates = _read_author_ts_coordinate_fixture()
                if (
                    provided["initial_transition_state_structures"]
                    != expected_coordinates
                ):
                    raise ContractValidationError("作者模式拒绝漂移的 TS 初始坐标")
                initial_transition_state_structures = copy.deepcopy(
                    provided["initial_transition_state_structures"]
                )
            if "initial_calculation_structures" in provided:
                if initial_calculation_structures is None:
                    raise ContractValidationError(
                        "当前作者模式不接受 initial_calculation_structures"
                    )
                if provided["initial_calculation_structures"] != initial_calculation_structures:
                    raise ContractValidationError("作者模式拒绝漂移的 Case 3 首点结构")
        retrieval_context = _require_object(raw["retrieval_context"], "retrieval_context")
        if set(retrieval_context) != {
            "retrieval_source",
            "chemistry_context_summary",
        } or retrieval_context["retrieval_source"] != "author_fixture":
            raise ContractValidationError("作者输入 retrieval_context 字段非法")
    hypothesis = _require_object(raw["hypothesis"], "hypothesis")
    if set(hypothesis) != {"hypothesis_id", "strategy_name", "reasoning"}:
        raise ContractValidationError("作者主假设必须只包含模型可见的三项正文")
    try:
        validate_structured_hypothesis_claims(hypothesis, allow_legacy=False)
    except ContractValidationError as exc:
        raise ContractValidationError("作者主假设正文不满足结构化假设合同") from exc
    if minimal_context_mode:
        selection = {}
    else:
        selection = _require_object(raw["selection"], "selection")
        if set(selection) != {
            "selection_status",
            "selection_method",
            "selection_reason",
        } or selection["selection_status"] != "selected":
            raise ContractValidationError("作者主假设 selection 事实非法")
    provenance = _require_object(raw["provenance"], "provenance")
    expected_provenance_fields = {
        "author_source_document_id",
        "primary_pdf_sha256",
        "primary_pages",
        "independent_author_complement_pdf_sha256",
    }
    if definition.published_artifact_sha256 is not None:
        expected_provenance_fields.add("published_execution_artifact_sha256")
    if set(provenance) != expected_provenance_fields:
        raise ContractValidationError("作者主假设 provenance 字段非法")
    if (
        provenance["primary_pdf_sha256"] != definition.primary_pdf_sha256
        or provenance["independent_author_complement_pdf_sha256"]
        != definition.complement_pdf_sha256
        or provenance["primary_pages"] != definition.primary_pages
        or (
            definition.published_artifact_sha256 is not None
            and provenance["published_execution_artifact_sha256"]
            != definition.published_artifact_sha256
        )
    ):
        raise ContractValidationError("作者主假设 provenance 哈希或页码不匹配")
    fixture_sha256 = _canonical_sha256(raw)
    return AuthorMainHypothesisFixture(
        fixture_id=definition.fixture_id,
        fixture_sha256=fixture_sha256,
        case_profile=definition.case_profile,
        question_sha256=question_sha256,
        calculation_unit_sha256=calculation_unit_sha256,
        calculation_unit=calculation_unit,
        initial_transition_state_structures=initial_transition_state_structures,
        initial_calculation_structures=initial_calculation_structures,
        retrieval_context=retrieval_context,
        hypothesis=hypothesis,
        selection=selection,
        provenance=provenance,
        author_mode=definition.author_mode,
        initial_query_angles=definition.initial_query_angles,
        constraint_semantics=(
            build_case3_advisory_constraint_semantics()
            if definition.author_mode == CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE
            else None
        ),
    )


def load_case1_author_main_hypothesis(
    scientific_question: str,
    supplementary_scientific_input: Mapping[str, Any] | None = None,
) -> AuthorMainHypothesisFixture:
    """兼容 Case 1 调用方，加载 Case 1 作者主假设夹具。"""

    return load_author_main_hypothesis(
        scientific_question,
        author_mode=CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE,
        supplementary_scientific_input=supplementary_scientific_input,
    )


def build_author_fixture_retrieval_result(
    fixture: AuthorMainHypothesisFixture,
) -> dict[str, Any]:
    """把作者夹具的冻结上下文整理成 RetrievalCompleted 可保存正文。"""

    if fixture.author_mode in _CASE3_MINIMAL_CONTEXT_MODES:
        return {}
    context_summary = fixture.retrieval_context["chemistry_context_summary"]
    if fixture.constraint_semantics is not None:
        context_summary = (
            "Case 3 四种配体模型的初始结构、电子态和坐标索引是不可变事实；"
            "作者方法和扫描参数是 advisory reference，可由模型调整并说明可比性影响。"
        )
    chemistry_context = {
        "retrieval_source": "author_fixture",
        "calculation_unit": copy.deepcopy(dict(fixture.calculation_unit)),
        "context_summary": context_summary,
    }
    if fixture.initial_transition_state_structures is not None:
        chemistry_context["initial_transition_state_structures"] = copy.deepcopy(
            dict(fixture.initial_transition_state_structures)
        )
    if fixture.initial_calculation_structures is not None:
        chemistry_context["initial_calculation_structures"] = copy.deepcopy(
            dict(fixture.initial_calculation_structures)
        )
    if fixture.constraint_semantics is not None:
        chemistry_context["constraint_semantics"] = copy.deepcopy(
            dict(fixture.constraint_semantics)
        )
    return {
        "retrieval_source": "author_fixture",
        "chemistry_context": chemistry_context,
        "retrieved_papers": [],
        "paper_search_status": {
            "mode": "author_fixture_import",
            "target_reached": True,
            "actual_papers": 0,
        },
    }


def write_author_hypothesis_import_manifest(
    task_root: str | Path,
    fixture: AuthorMainHypothesisFixture,
    scientific_question: str,
    operation_instance_id: str,
) -> str:
    """在任务目录内原子写入作者夹具摘要和诚实导入 manifest。

    输入是任务根、已校验的作者夹具、题面和操作身份；动作把固定夹具及导入摘要
    原子写入任务目录，不读取计算结果。返回 manifest 的任务相对引用；题面摘要或
    任务目录不一致时抛出 ContractValidationError。
    """

    if hashlib.sha256(scientific_question.encode("utf-8")).hexdigest() != fixture.question_sha256:
        raise ContractValidationError("作者导入 manifest 的题面摘要与夹具不一致")
    fixture_payload = _read_fixture_json(
        author_main_hypothesis_definition(fixture.author_mode)
    )
    fixture_ref = write_content_addressed_json(
        task_root,
        _AUTHOR_MANIFEST_DIRECTORY,
        "fixture",
        fixture_payload,
    )
    manifest = _build_author_import_manifest(
        fixture=fixture,
        fixture_ref=fixture_ref,
        operation_instance_id=operation_instance_id,
    )
    return write_content_addressed_json(
        task_root,
        _AUTHOR_MANIFEST_DIRECTORY,
        "manifest",
        manifest,
    )


def _build_author_import_manifest(
    *,
    fixture: AuthorMainHypothesisFixture,
    fixture_ref: str,
    operation_instance_id: str,
) -> dict[str, Any]:
    """构造作者导入 manifest 的唯一公开正文。"""

    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "schema_version": 1,
        "operation": "author_hypothesis_import",
        "operation_instance_id": operation_instance_id,
        "expected_combination_count": 1,
        "fixture_id": fixture.fixture_id,
        "fixture_sha256": fixture.fixture_sha256,
        "fixture_ref": fixture_ref,
        "question_sha256": fixture.question_sha256,
        "provenance": copy.deepcopy(dict(fixture.provenance)),
        "model_calls": [],
        "generation_called": False,
    }
    if fixture.author_mode not in _CASE3_MINIMAL_CONTEXT_MODES:
        manifest.update(
            {
                "calculation_unit_sha256": fixture.calculation_unit_sha256,
                "selection_status": fixture.selection["selection_status"],
                "selection_method": fixture.selection["selection_method"],
            }
        )
    return manifest


def _validate_author_import_manifest(
    task_root: str | Path,
    manifest_ref: Any,
    *,
    fixture: AuthorMainHypothesisFixture,
    operation_instance_id: str,
) -> None:
    """核对既有作者 manifest、fixture hash/ref 和冻结正文。"""

    fixture_payload = _read_fixture_json(
        author_main_hypothesis_definition(fixture.author_mode)
    )
    expected_fixture_ref = _content_addressed_ref(fixture_payload, "fixture")
    if manifest_ref != _content_addressed_ref(
        _build_author_import_manifest(
            fixture=fixture,
            fixture_ref=expected_fixture_ref,
            operation_instance_id=operation_instance_id,
        ),
        "manifest",
    ):
        raise ContractValidationError("作者 BatchCreated 的 manifest 引用与冻结内容不一致")
    manifest = _read_content_addressed_json(
        task_root,
        manifest_ref,
        "作者导入 manifest",
    )
    expected_manifest = _build_author_import_manifest(
        fixture=fixture,
        fixture_ref=expected_fixture_ref,
        operation_instance_id=operation_instance_id,
    )
    if canonical_json(manifest) != canonical_json(expected_manifest):
        raise ContractValidationError("作者导入 manifest 与冻结 fixture 摘要或正文不一致")
    fixture_from_artifact = _read_content_addressed_json(
        task_root,
        manifest["fixture_ref"],
        "作者导入 fixture_ref",
    )
    if canonical_json(fixture_from_artifact) != canonical_json(fixture_payload):
        raise ContractValidationError("作者导入 fixture_ref 内容与固定 fixture 不一致")


def build_author_hypothesis_batch_payload(
    task_id: str,
    batch_id: str,
    scientific_question: str,
    retrieval_result: Mapping[str, Any],
    fixture: AuthorMainHypothesisFixture,
) -> dict[str, Any]:
    """构造只含一条作者候选的正式 BatchCreated 正文。"""

    hypothesis = copy.deepcopy(dict(fixture.hypothesis))
    return {
        "batch_id": batch_id,
        "batch_index": 1,
        "operation_instance_id": (
            f"author-hypothesis-import-{task_id}-{fixture.fixture_sha256[:24]}"
        ),
        "question_snapshot": {
            "scientific_question": scientific_question,
            "initial_query_angles": list(fixture.initial_query_angles),
            "initial_retrieval_phase": copy.deepcopy(dict(retrieval_result)),
        },
        "generation_source": {
            "kind": "initial",
            "previous_batch_id": None,
            "source_record_ids": [],
        },
        "hypothesis_phase": {"ranked_strategies": [hypothesis]},
        "candidates": [
            {
                "hypothesis_id": hypothesis["hypothesis_id"],
                "candidate_index": 1,
                "rank": 1,
                "selection_status": "selected",
                "combination_id": f"combination:{task_id}:{batch_id}:1",
                "combination_index": 1,
            }
        ],
        "causal_source_ref": None,
        "investigation_protocol_version": 8,
    }


def import_author_main_hypothesis_batch(
    *,
    service: Any,
    task_id: str,
    task_root: str | Path,
    scientific_question: str,
    supplementary_scientific_input: Mapping[str, Any] | None = None,
    author_mode: str | None = None,
) -> Mapping[str, Any]:
    """幂等保存作者 RetrievalCompleted、导入 manifest 和单候选 BatchCreated。

    输入必须来自已创建的 author-mode TaskCreated 和固定题面；动作先核对任务模式与
    夹具，再复用或保存冻结 Retrieval 事实、内容寻址 manifest 和单一批次。副作用只
    写当前 Run Bundle/Working Memory；重复调用读取既有事件，不重新导入、不调用任何
    agent。任务事实、夹具、题面或上下文不一致时 fail-closed。
    """

    task = service.get_task_created(task_id)
    task_metadata = task.get("hypothesis_import") if isinstance(task, Mapping) else None
    if not isinstance(task, Mapping) or not isinstance(task_metadata, Mapping) or not is_author_main_hypothesis_mode(
        task_metadata.get("mode")
    ):
        raise ContractValidationError("当前 TaskCreated 未冻结作者主假设实验模式")
    metadata = task_metadata
    frozen_author_mode = metadata["mode"]
    if author_mode is not None and author_mode != frozen_author_mode:
        raise ContractValidationError("作者导入调用模式与 TaskCreated 不一致")
    fixture = load_author_main_hypothesis(
        scientific_question,
        author_mode=frozen_author_mode,
        supplementary_scientific_input=supplementary_scientific_input,
    )
    expected_metadata = {
        "mode": fixture.author_mode,
        "fixture_id": fixture.fixture_id,
        "fixture_sha256": fixture.fixture_sha256,
        "manifest_schema": _MANIFEST_SCHEMA,
        "expected_combination_count": 1,
    }
    if fixture.author_mode == CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE:
        structure_input = metadata.get("structure_input")
        expected_metadata["structure_input"] = validate_case3_prepared_input_metadata(
            structure_input
        )
    if canonical_json(dict(metadata)) != canonical_json(expected_metadata):
        raise ContractValidationError("TaskCreated 作者夹具摘要与固定 fixture 不一致")
    if task.get("scientific_question") != scientific_question:
        raise ContractValidationError("TaskCreated scientific_question 与作者夹具题面不一致")
    task_config = task.get("task_config")
    if not isinstance(task_config, Mapping):
        raise ContractValidationError("TaskCreated 缺少冻结作者 task_config")
    if (
        task_config.get("investigation_mode") != fixture.author_mode
        or task_config.get("investigation_combination_count") != 1
        or task_config.get("top_n_hypotheses") != 1
    ):
        raise ContractValidationError("TaskCreated 作者 mode/count 与固定单组合实验不一致")
    task_context = task.get("chemistry_context")
    if fixture.author_mode in _CASE3_MINIMAL_CONTEXT_MODES:
        if task_context != {}:
            raise ContractValidationError(
                "Case 3 精简主假设 TaskCreated 不得携带结构化科学上下文"
            )
    else:
        task_unit = (
            task_context.get("calculation_unit")
            if isinstance(task_context, Mapping)
            else None
        )
        if task_unit != fixture.calculation_unit:
            raise ContractValidationError("TaskCreated chemistry_context 与作者输入不一致")
    task_structures = (
        task_context.get("initial_transition_state_structures")
        if isinstance(task_context, Mapping)
        else None
    )
    if fixture.initial_transition_state_structures is None:
        if task_structures not in (None, {}):
            raise ContractValidationError("TaskCreated 不应携带未冻结的 TS 初始坐标")
    elif task_structures != fixture.initial_transition_state_structures:
        raise ContractValidationError("TaskCreated TS 初始坐标与作者夹具不一致")
    task_calculation_structures = (
        task_context.get("initial_calculation_structures")
        if isinstance(task_context, Mapping)
        else None
    )
    if fixture.initial_calculation_structures is None:
        if task_calculation_structures not in (None, {}):
            raise ContractValidationError("TaskCreated 不应携带未冻结的普通计算初始结构")
    elif task_calculation_structures != fixture.initial_calculation_structures:
        raise ContractValidationError("TaskCreated 普通计算初始结构与作者夹具不一致")
    retrieval_result = build_author_fixture_retrieval_result(fixture)
    existing_retrieval = service.get_retrieval_completed(task_id)
    if existing_retrieval is None:
        service.record_retrieval_completed(task_id, retrieval_result)
    elif existing_retrieval.get("retrieval_result") != retrieval_result:
        raise ContractValidationError("既有 RetrievalCompleted 与作者夹具不一致")
    existing_batch = service.find_latest_batch(task_id)
    if isinstance(existing_batch, Mapping):
        batch_id = existing_batch.get("batch_id")
        expected_batch_id = f"batch:{task_id}:1"
        if batch_id != expected_batch_id:
            raise ContractValidationError("既有作者批次身份与固定首批不一致")
        expected_payload = build_author_hypothesis_batch_payload(
            task_id,
            expected_batch_id,
            scientific_question,
            retrieval_result,
            fixture,
        )
        receipt_reader = getattr(service, "get_batch_created_receipt", None)
        if not callable(receipt_reader):
            raise ContractValidationError("作者批次恢复缺少 BatchCreated receipt 读取接口")
        receipt = receipt_reader(task_id, expected_batch_id)
        if not isinstance(receipt, Mapping):
            raise ContractValidationError("既有作者 BatchCreated receipt 必须是 object")
        payload = receipt.get("payload")
        if not isinstance(payload, Mapping) or canonical_json(dict(payload)) != canonical_json(
            expected_payload
        ):
            raise ContractValidationError(
                "既有作者 BatchCreated 与固定 fixture hypothesis/combination/selection 不一致"
            )
        source_refs = receipt.get("source_refs")
        expected_operation_instance_id = expected_payload["operation_instance_id"]
        expected_fixture_ref = _content_addressed_ref(
            _read_fixture_json(author_main_hypothesis_definition(fixture.author_mode)),
            "fixture",
        )
        expected_manifest_ref = _content_addressed_ref(
            _build_author_import_manifest(
                fixture=fixture,
                fixture_ref=expected_fixture_ref,
                operation_instance_id=expected_operation_instance_id,
            ),
            "manifest",
        )
        if source_refs != [expected_manifest_ref]:
            raise ContractValidationError(
                "既有作者 BatchCreated 的 manifest fixture_ref 不符合冻结引用"
            )
        _validate_author_import_manifest(
            task_root,
            expected_manifest_ref,
            fixture=fixture,
            operation_instance_id=expected_operation_instance_id,
        )
        return dict(payload)
    batch_id = f"batch:{task_id}:1"
    manifest_ref = write_author_hypothesis_import_manifest(
        task_root,
        fixture,
        scientific_question,
        (
            f"author-hypothesis-import-{task_id}-{fixture.fixture_sha256[:24]}"
        ),
    )
    payload = build_author_hypothesis_batch_payload(
        task_id,
        batch_id,
        scientific_question,
        retrieval_result,
        fixture,
    )
    service.create_hypothesis_batch(task_id, payload, [manifest_ref])
    return service.get_batch_created(task_id, batch_id)


__all__ = [
    "AUTHOR_MAIN_HYPOTHESIS_DEFINITIONS",
    "CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE",
    "CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE",
    "CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE",
    "CASE3_AUTHOR_MAIN_HYPOTHESIS_ONLY_MODE",
    "CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE",
    "CASE3_PREPARED_INPUT_SCHEMA",
    "CASE3_PREPARED_INPUT_FILENAMES",
    "AuthorMainHypothesisFixture",
    "AuthorMainHypothesisDefinition",
    "author_main_hypothesis_definition",
    "author_main_hypothesis_mode_for_case_profile",
    "build_author_fixture_retrieval_result",
    "build_case3_advisory_constraint_semantics",
    "build_case3_workspace_input_paths",
    "build_author_hypothesis_batch_payload",
    "import_author_main_hypothesis_batch",
    "is_author_main_hypothesis_mode",
    "load_case3_prepared_input_bundle",
    "load_author_main_hypothesis",
    "load_case1_author_main_hypothesis",
    "write_author_hypothesis_import_manifest",
    "write_case3_prepared_input_files",
    "materialize_case3_session_input_files",
    "normalize_author_hypotheses",
    "validate_case3_prepared_input_metadata",
]
