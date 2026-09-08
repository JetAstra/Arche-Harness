"""生成、归并并排序科学假设。"""

import json
import re
import logging
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
    Literal,
)
from itertools import combinations

from chemistry_multiagent.utils.model_operation import (
    DynamicModelOperationRunner,
    ModelOperationRunner,
    ModelOperationStageError,
    ModelStageObserver,
    ModelStage,
)
from chemistry_multiagent.memory.contracts import (
    derive_hypothesis_compatibility_text,
)
from chemistry_multiagent.memory.working_memory_validation import (
    validate_investigation_working_memory_payload,
)
from chemistry_multiagent.memory.projections import project_hypothesis_regeneration_context

from chemistry_multiagent.utils.llm_api import call_compiled_model_request
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_chemical_capability_projection,
    project_minichem_stage_tool_projection,
    safe_join_task_path,
)
from chemistry_multiagent.utils.model_run_observer import ModelRunStageObserver
from chemistry_multiagent.utils.model_run_view_publisher import ModelRunViewPublisher

logger = logging.getLogger(__name__)

# 英文 Query 继续使用历史词数下限；中文或中英混合 Query 使用非空白字符和
# CJK 字符下限，避免短标签绕过科学详细度合同。
_QUERY_MIN_NON_WHITESPACE_CHARS = 80
_QUERY_MIN_CJK_CHARS = 40
_QUERY_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SOURCE_QUERY_ANGLES_FIELD = "source_query_angles"


# ==================== 假设协议错误 ====================

class HypothesisProtocolError(RuntimeError):
    """模型连续多次没有给出可用结果时，终止本批假设生成。

    异常会保留失败发生在哪一步以及失败类型，Controller 后续可以据此暂停
    或重试整次操作，但不能用默认假设悄悄替代模型结果。
    """

    def __init__(
        self,
        stage: str,
        failure_kind: str,
        query_index: int | None = None,
    ):
        self.stage = stage
        self.failure_kind = failure_kind
        self.query_index = query_index
        query_suffix = (
            f" Query {query_index}"
            if isinstance(query_index, int) and query_index > 0
            else ""
        )
        super().__init__(f"{stage}{query_suffix} stage 失败：{failure_kind}")


_HYPOTHESIS_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "strategy_name": {"type": "string", "minLength": 1},
        "reasoning": {"type": "string", "minLength": 1, "maxLength": 3200},
    },
    "required": ["strategy_name", "reasoning"],
}


_RELATIONSHIP_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relationship": {
            "type": "string",
            "enum": ["merge", "complement", "conflict", "independent"],
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 2400},
    },
    "required": ["relationship", "reason"],
}


_SYNTHESIS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    **copy.deepcopy(_HYPOTHESIS_ITEM_SCHEMA),
}


_RANK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "analysis": {"type": "string", "minLength": 1, "maxLength": 3200},
        "reasoning": {"type": "string", "minLength": 1, "maxLength": 2400},
        "winner": {"type": "string", "enum": ["A", "B"]},
        "loser": {"type": "string", "enum": ["A", "B"]},
    },
    "required": ["analysis", "reasoning", "winner", "loser"],
}


# ==================== 假设输出格式 ====================

def _hypothesis_array_schema(expected_count: int) -> Dict[str, Any]:
    """按本次任务要求的数量生成假设输出格式。

    数量直接写进校验规则，因此模型多给或少给一项都会被拒绝，避免后续
    排名阶段面对来源不明的候选数量。
    """

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        "minItems": expected_count,
        "maxItems": expected_count,
        "items": copy.deepcopy(_HYPOTHESIS_ITEM_SCHEMA),
    }


_MERGED_HYPOTHESIS_SCHEMA = copy.deepcopy(_SYNTHESIS_SCHEMA)


def _validate_relationship_payload(value: Any) -> Dict[str, str]:
    """校验关系 JSON，并规范化审计理由。"""

    if not isinstance(value, Mapping) or set(value) != {"relationship", "reason"}:
        raise ValueError("relationship 输出必须恰含 relationship 与 reason")
    relationship = value["relationship"]
    reason = value["reason"]
    if relationship not in {"merge", "complement", "conflict", "independent"}:
        raise ValueError("relationship 必须是四类注册关系之一")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("relationship.reason 必须是非空字符串")
    if len(reason.strip()) > 2400:
        raise ValueError("relationship.reason 超过 2400 字符")
    return {"relationship": relationship, "reason": reason.strip()}


def _validate_synthesis_payload(value: Any) -> Dict[str, str]:
    """校验 synthesis 的两字段科学正文。"""

    if not isinstance(value, Mapping) or set(value) != {"strategy_name", "reasoning"}:
        raise ValueError("synthesis 输出必须恰含 strategy_name 与 reasoning")
    normalized = {}
    for field, limit in (("strategy_name", 600), ("reasoning", 3200)):
        text = value[field]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"synthesis.{field} 必须是非空字符串")
        if len(text.strip()) > limit:
            raise ValueError(f"synthesis.{field} 超过 {limit} 字符")
        normalized[field] = text.strip()
    return normalized


class _CalculationUnitSynthesisValidator:
    """为 synthesis 保存固定计算单元，并提供可恢复的等价校验器。"""

    def __init__(self, calculation_unit: Mapping[str, Any]):
        self._calculation_unit = copy.deepcopy(dict(calculation_unit))

    def __call__(self, value: Any) -> Dict[str, str]:
        normalized = _validate_synthesis_payload(value)
        HypothesisAgent._validate_calculation_unit_hypothesis_scope(
            [normalized],
            self._calculation_unit,
        )
        return normalized

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _CalculationUnitSynthesisValidator) and (
            self._calculation_unit == other._calculation_unit
        )


class _CalculationUnitHypothesisBatchValidator:
    """为 hypothesis/regeneration 保存同一冻结 calculation_unit。"""

    def __init__(self, calculation_unit: Mapping[str, Any]):
        self._calculation_unit = copy.deepcopy(dict(calculation_unit))

    def __call__(self, value: Any) -> List[Dict[str, Any]]:
        return HypothesisAgent._normalize_generated_hypotheses(
            value,
            calculation_unit=self._calculation_unit,
        )

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _CalculationUnitHypothesisBatchValidator) and (
            self._calculation_unit == other._calculation_unit
        )


def _validate_rank_payload(value: Any) -> Dict[str, str]:
    """校验排名比较的完整审计对象，并确保胜负标签相反。"""

    if not isinstance(value, Mapping) or set(value) != {
        "analysis",
        "reasoning",
        "winner",
        "loser",
    }:
        raise ValueError("rank 输出必须恰含 analysis、reasoning、winner、loser")
    analysis = value["analysis"]
    reasoning = value["reasoning"]
    winner = value["winner"]
    loser = value["loser"]
    if not isinstance(analysis, str) or not analysis.strip():
        raise ValueError("rank.analysis 必须是非空字符串")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("rank.reasoning 必须是非空字符串")
    if winner not in {"A", "B"} or loser not in {"A", "B"} or winner == loser:
        raise ValueError("rank.winner 与 rank.loser 必须是相反的 A/B")
    return {
        "analysis": analysis.strip(),
        "reasoning": reasoning.strip(),
        "winner": winner,
        "loser": loser,
    }


def _read_consumed_hypothesis_ids(
    ranked: Sequence[Mapping[str, Any]],
) -> set[str]:
    """读取完整排名中由 synthesis_sources 消费的原始父项身份。"""

    consumed_ids: set[str] = set()
    for hypothesis in ranked:
        sources = hypothesis.get("synthesis_sources")
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            parent_ids = source.get("parent_hypothesis_ids")
            if not isinstance(parent_ids, list):
                continue
            consumed_ids.update(
                parent_id
                for parent_id in parent_ids
                if isinstance(parent_id, str) and parent_id
            )
    return consumed_ids


_STAGE_REGENERATION_PACKAGE_KEYS = {
    "scientific_context",
    "current_query",
    "prior_hypothesis_outcomes",
    "next_batch_index",
    "generation_config",
}
_STAGE_REGENERATION_OPTIONAL_INTERNAL_KEYS = {
    "source_record_ids",
    # Investigation bridge 传入的完整、已脱敏 Working Memory；它不是事件身份，
    # 但必须保留给下一批 Hypothesis 模型，避免只把终局 marker 当作历史。
    "investigation_working_memory",
}
_REGENERATION_CONTEXT_KEYS = {
    "scientific_question",
    "initial_query_angles",
    "initial_retrieval_phase",
    "chemistry_context",
}

_PUBLIC_REGENERATION_PRIVATE_KEYS = {
    "workspace",
    "workspace_ref",
    "workspace_path",
    "scope_id",
    "thread_id",
    "thread",
    "codex_thread_id",
    "review_thread",
    "review_thread_id",
    "review_turn",
    "review_turn_id",
    "turn",
    "turn_id",
    "transcript",
    "path",
    "paths",
    "manifest_path",
    "resume_state_path",
    "source_ref",
    "source_refs",
    "event_id",
    "record_id",
    "api_key",
    "access_token",
    "secret",
    "password",
    "credential",
    "credentials",
}
_PUBLIC_REGENERATION_PRIVATE_KEY_NORMALIZED = {
    re.sub(r"[^a-z0-9]", "", item.casefold())
    for item in _PUBLIC_REGENERATION_PRIVATE_KEYS
}
_PUBLIC_REGENERATION_PRIVATE_TEXT_RE = re.compile(
    r"\b(?:workspace_ref|workspace_path|scope_id|codex_thread_id|thread_id|"
    r"review_thread_id|review_turn_id|transcript)\b",
    re.IGNORECASE,
)
_PUBLIC_REGENERATION_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|\s)(?:/|[A-Za-z]:[\\/]|file://)",
    re.IGNORECASE,
)
_PUBLIC_REGENERATION_RELATIVE_PATH_RE = re.compile(
    r"(?:^|[\s(])(?:\.\.?/|(?:local_data|outputs|working_memory|\.runtime|"
    r"run_bundle|tmp|home|workspace|scope)/)",
    re.IGNORECASE,
)
_PUBLIC_REGENERATION_SECRET_RE = re.compile(
    r"(?:api[_ -]?key|access[_ -]?token|secret|password)\s*=|\bsk-[A-Za-z0-9_-]{8,}\b",
    re.IGNORECASE,
)
_PUBLIC_REGENERATION_FORBIDDEN_KEYS = {
    "schema",
    "schemaversion",
    "task",
    "taskid",
    "taskroot",
    "batch",
    "batchid",
    "batchindex",
    "operation",
    "operationid",
    "operationinstanceid",
    "operationkey",
}


def _validate_public_regeneration_string(item: str, path: str) -> None:
    """检查公共 package 字符串，拒绝会话身份、路径和凭据。"""

    if _PUBLIC_REGENERATION_PRIVATE_TEXT_RE.search(item):
        raise ValueError(f"{path} 不能包含私有会话身份")
    if _PUBLIC_REGENERATION_ABSOLUTE_PATH_RE.search(item):
        raise ValueError(f"{path} 不能包含宿主路径或 file URI")
    if _PUBLIC_REGENERATION_RELATIVE_PATH_RE.search(item):
        raise ValueError(f"{path} 不能包含宿主相对路径")
    if _PUBLIC_REGENERATION_SECRET_RE.search(item):
        raise ValueError(f"{path} 不能包含凭据")


def _validate_public_regeneration_mapping(
    item: Mapping[str, Any],
    path: str,
    *,
    root: bool,
) -> None:
    """检查公共 package object 的字段名并递归检查字段值。"""

    for key, nested in item.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path} 的 object key 非法")
        normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
        if not root and (
            normalized in _PUBLIC_REGENERATION_PRIVATE_KEY_NORMALIZED
            or normalized in _PUBLIC_REGENERATION_FORBIDDEN_KEYS
        ):
            raise ValueError(f"{path}.{key} 是未注册的私有或内部字段")
        # Working Memory 有独立 schema 和身份校验，公共递归不重复展开其内部字段。
        if root and key == "investigation_working_memory":
            continue
        _walk_public_regeneration_package_json(nested, f"{path}.{key}")


def _validate_public_regeneration_list(item: list[Any], path: str) -> None:
    """检查公共 package 数组并递归检查每个数组元素。"""

    for index, nested in enumerate(item):
        _walk_public_regeneration_package_json(nested, f"{path}[{index}]")


def _walk_public_regeneration_package_json(
    item: Any,
    path: str,
    *,
    root: bool = False,
) -> None:
    """按 JSON 类型分派公共 package 的递归校验动作。"""

    if isinstance(item, str):
        _validate_public_regeneration_string(item, path)
        return
    if isinstance(item, Mapping):
        _validate_public_regeneration_mapping(item, path, root=root)
        return
    if isinstance(item, list):
        _validate_public_regeneration_list(item, path)
        return
    if item is None or isinstance(item, (bool, int, float)):
        return
    raise ValueError(f"{path} 不是严格 JSON 值")


def _validate_public_regeneration_package_json(
    value: Any,
    *,
    label: str,
) -> None:
    """递归检查旧 Hypothesis 公共 package，阻止私有身份、路径和凭据下沉。"""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 必须是严格 JSON") from exc
    _walk_public_regeneration_package_json(payload, label, root=True)



def _project_calculation_unit_context(value: Any) -> Dict[str, Any]:
    """从科学上下文复制可选计算单元，缺失或类型错误时返回空对象。

    函数不修改输入，也不在此处重复执行控制层校验。
    """

    source = value if isinstance(value, Mapping) else {}
    unit = source.get("calculation_unit")
    if not isinstance(unit, Mapping) or not unit:
        return {}
    return {"calculation_unit": copy.deepcopy(dict(unit))}


_CALCULATION_UNIT_QUERY_NUMERIC_PATTERNS = (
    re.compile(
        r"(?:\b(?:within|below|above|under|over|less\s+than|greater\s+than|"
        r"at\s+least|at\s+most|no\s+(?:more|less)\s+than|not\s+(?:more|less)\s+than|"
        r"between|approximately|approx\.?|about|around|roughly|exceed(?:s|ing)?|"
        r"threshold|cutoff|tolerance|margin|range|criterion|criteria|acceptance|"
        r"expected|should\s+be|must\s+be)\b|"
        r"(?:<=|>=|≤|≥|≈|±|~))\s*(?:the\s+|an?\s+|a\s+)?"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+|zero|one|two|three|four|five|six|seven|"
        r"eight|nine|ten)(?:\s*(?:i|j|%|[A-Za-zÅμµ°]+(?:/[A-Za-zÅμµ°]+)?))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:rmsd|tolerance|threshold|cutoff|deviation|error|difference|"
        r"accuracy|precision|distance|barrier|energy\s+(?:difference|gap)|"
        r"frequency(?:\s+value)?|imaginary\s+frequency)\b[^\n<>]{0,40}"
        r"(?:<=|>=|<|>|≤|≥|≈|±|~)\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:rmsd|tolerance|threshold|cutoff|deviation|error|difference|"
        r"accuracy|precision|distance|barrier|energy\s+(?:difference|gap)|"
        r"frequency(?:\s+value)?|imaginary\s+frequency)\b[^\n<>]{0,24}"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:Å|angstroms?|a\.u\.|kcal(?:ories)?\s*/\s*mol|kJ\s*/\s*mol|"
        r"cm\s*[-−]?\s*1|%)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:Å|angstroms?|a\.u\.|kcal(?:ories)?\s*/\s*mol|kJ\s*/\s*mol|"
        r"cm\s*[-−]?\s*1|%)\s*"
        r"(?:rmsd|tolerance|threshold|cutoff|deviation|error|difference|"
        r"accuracy|precision|distance|barrier|frequency|energy\s+(?:difference|gap))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:小于|大于|低于|高于|不超过|不低于|至少|至多|约为?|阈值|容差|"
        r"误差|范围|界限|精度)\s*(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)|"
        r"零|一|二|三|四|五|六|七|八|九|十)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:至|到|[-–—~])\s*"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:%|[A-Za-zÅμµ°]+(?:/[A-Za-zÅμµ°]+)?)",
        re.IGNORECASE,
    ),
)


_CALCULATION_UNIT_QUERY_EXTERNAL_PATTERN = re.compile(
    r"\b(?:benchmark(?:s)?|literature|published|high[- ]level|higher[- ]level|"
    r"gold[- ]standard)\b|"
    r"\b(?:external|experiment(?:al(?:ly)?)?|reported|known)\s+"
    r"(?:benchmark(?:s)?|baseline(?:s)?|data|value|result|measurement|evidence|"
    r"reference(?:\s+(?:data|value|result))?|comparison)\b|"
    r"文献|基准|(?:实验|外部|报道|已知)(?:数据|值|结果|基准|证据|参考)",
    re.IGNORECASE,
)


_CALCULATION_UNIT_QUERY_COMPETITION_PATTERN = re.compile(
    r"\b(?:competitive|competition|competing|alternative|parallel)\s+"
    r"(?:path(?:way|s)?|route|reaction|transition\s+state|mechanism)|"
    r"\b(?:selectiv(?:ity|e)|enantioselectiv(?:ity|e)|enantiomeric|"
    r"stereoselectiv(?:ity|e)|diastereoselectiv(?:ity|e)|diastereomer(?:ic)?|"
    r"ee|e\.e\.|major/minor|major\s+and\s+minor|re\s*/\s*si|r\s*/\s*s|"
    r"delta[-\s]*delta[-\s]*g|ddg)\b|"
    r"Δ\s*Δ\s*G|竞争(?:路径|反应|机理)?|对映(?:选择性)?|立体选择性|"
    r"非对映选择性|选择性|ΔΔG",
    re.IGNORECASE,
)


_CALCULATION_UNIT_QUERY_INITIAL_GUESS_METHOD_PATTERN = re.compile(
    r"\b(?:LST|NEB|QST(?:2|3)?|GSM)\b|"
    r"\blinear\s+synchronous\s+transit\b|"
    r"\bnudged\s+elastic\s+band\b|"
    r"\bstring\s+(?:method|approach|path|initial[- ]guess)\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_QUERY_UNDECLARED_CRITERION_PATTERN = re.compile(
    r"\b(?:threshold|tolerance|cutoff|acceptance\s+criteria?|"
    r"convergence\s+(?:criteria?|standard|standards)|"
    r"default\s+(?:program|solver|convergence)\s+(?:criteria?|standard|standards|threshold)|"
    r"(?:program\s+)?(?:default|typical|standard|usual|reasonable)\s+"
    r"(?:program\s+|solver\s+)?(?:bond[- ]length|convergence(?:/rmsd)?|rmsd|"
    r"distance)(?:\s+(?:criteria?|standard|standards|threshold))?)\b|"
    r"(?:阈值|容差|验收标准|收敛(?:标准|判据)|默认(?:程序|软件|求解器)?(?:收敛)?(?:标准|判据|阈值))",
    re.IGNORECASE,
)


_CALCULATION_UNIT_QUERY_THERMOCHEMISTRY_PATTERN = re.compile(
    r"\b(?:thermochem(?:istry|ical)?|thermodynamic(?:s)?|"
    r"gibbs\s+free\s+energy|free[- ]energy|zero[- ]point(?:\s+energy)?|"
    r"thermal\s+correction(?:s)?|boltzmann(?:\s+population)?|"
    r"thermochemical\s+data)\b|"
    r"热化学|热力学|吉布斯(?:自由能)?|自由能|零点能|热修正|玻尔兹曼",
    re.IGNORECASE,
)


_CALCULATION_UNIT_QUERY_NEGATED_SCOPE_PATTERN = re.compile(
    r"(?:\bwithout\b|\b(?:no|not|never|neither|nor)\b(?!\s+(?:only|just|merely)\b)|"
    r"\b(?:do|does|did|should|must)\s+not\b(?!\s+(?:only|just|merely)\b)|"
    r"\b(?:exclude|excluding|omit|omitting|avoid|avoiding)\b)"
    r"(?:\s+[^\s.;:!?<>]+){0,2}\s*$",
    re.IGNORECASE,
)


_CALCULATION_UNIT_QUERY_CONFORMER_PATTERN = re.compile(
    r"\b(?:conformer(?:s)?|conformational(?:\s+ensemble)?|"
    r"rotamer(?:s)?|isomer(?:s)?|cis/trans|E[-–/]Z|"
    r"protonat(?:ed|ion)|tautomer(?:s)?)\b|"
    r"构象(?:系综|集合|采样)?|异构(?:体|形式)?|顺反|质子化|互变异构",
    re.IGNORECASE,
)


_CALCULATION_UNIT_SCOPE_DECLARATION_FIELDS = frozenset(
    {
        "analysis",
        "analyses",
        "analysis_type",
        "analysis_types",
        "job",
        "jobs",
        "job_type",
        "job_types",
    }
)


_CALCULATION_UNIT_HYPOTHESIS_SCOPE_DETAIL_PREFIX = "calculation_unit_scope.v1:"
_CALCULATION_UNIT_HYPOTHESIS_SCOPE_DETAIL_LIMIT = 8


_CALCULATION_UNIT_HYPOTHESIS_SCOPE_RULES = (
    (
        "undeclared_extra_object",
        re.compile(
            r"\b(?:additional|extra|another|new|added)\s+"
            r"(?:reactant|product|catalyst|counter[- ]ion|counterion|intermediate|"
            r"solvent|temperature|pathway|route)\b|\bTfOH\b",
            re.IGNORECASE,
        ),
    ),
    (
        "undeclared_thermochemistry",
        re.compile(
            r"\b(?:Boltzmann|thermochemical ensemble|thermochem(?:istry|ical)|"
            r"thermodynamic(?:s)?|Gibbs\s+free\s+energy|free[- ]energy|"
            r"zero[- ]point(?:\s+energy)?|thermal\s+correction(?:s)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "undeclared_structure_or_condition",
        re.compile(
            r"\b(?:temperature sensitivity|method robustness|functional comparison|"
            r"basis[- ]set screening)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "undeclared_scan",
        re.compile(
            r"\b(?:candidate_count|max_candidates|orientation_trials)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "undeclared_structure_or_condition",
        re.compile(
            r"\b(?:different|other|alternate|alternative)\s+"
            r"(?:solvent|temperature|method|basis|condition|approximation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "undeclared_structure_or_condition",
        re.compile(
            r"\b(?:replace|replac(?:ed|ing)|substitut(?:e|ed|ing))\b"
            r"[^\n,.;:!?]{0,120}\b(?:method|basis|functional|calculation|"
            r"solvent|condition|approximation|GFN|ALPB|xTB)\b|"
            r"\b(?:switch(?:ed|ing)?|instead of|rather than)\b"
            r"[^\n,.;:!?]{0,120}\b(?:method|basis|functional|calculation|"
            r"solvent|condition|approximation|GFN|ALPB|xTB)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "undeclared_external_or_competing_evidence",
        re.compile(
            r"\b(?:external|experimental|reported|literature|known)\s+"
            r"(?:benchmark|baseline|data|value|result|measurement|evidence|"
            r"reference|comparison)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "undeclared_external_or_competing_evidence",
        re.compile(
            r"\b(?:competitive|competition|competing|alternative|parallel)\s+"
            r"(?:path(?:way|s)?|route|reaction|transition\s+state|mechanism)|"
            r"\b(?:selectiv(?:ity|e)|enantioselectiv(?:ity|e)|enantiomeric|"
            r"stereoselectiv(?:ity|e)|diastereoselectiv(?:ity|e)|"
            r"diastereomer(?:ic)?|ee|e\.e\.|major/minor|major\s+and\s+minor|"
            r"re\s*/\s*si|r\s*/\s*s|delta[-\s]*delta[-\s]*g|ddg)\b|"
            r"Δ\s*Δ\s*G|竞争(?:路径|反应|机理)?|对映(?:选择性)?|立体选择性|"
            r"非对映选择性|选择性|ΔΔG",
            re.IGNORECASE,
        ),
    ),
)


_CALCULATION_UNIT_HYPOTHESIS_THERMOCHEMISTRY_SCOPE_PATTERN = next(
    pattern
    for rule, pattern in _CALCULATION_UNIT_HYPOTHESIS_SCOPE_RULES
    if rule == "undeclared_thermochemistry"
)


_CALCULATION_UNIT_HYPOTHESIS_CONFORMER_TOKEN_PATTERN = re.compile(
    r"\b(?:conformer(?:s)?|conformational|rotamer(?:s)?|isomer(?:s)?|"
    r"cis/trans|E[-–/]Z|protonat(?:ed|ion)|tautomer(?:s)?)\b|"
    r"构象(?:系综|集合|采样)?|异构(?:体|形式)?|顺反|质子化|互变异构",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_CONFORMER_ACTION_PATTERN = re.compile(
    r"\b(?:search(?:ing|ed)?|find(?:ing|s)?|locat(?:e|ing|ed)|"
    r"generat(?:e|ing|ed|ion)|build(?:ing)?|construct(?:ing|ed|ion)?|"
    r"creat(?:e|ing|ed|ion)|optimiz(?:e|ing|ed|ation)|relax(?:ed|ing|ation)?|"
    r"sampl(?:e|ing|ed)|enumerat(?:e|ing|ed|ion)|compar(?:e|ing|ed|ison)|"
    r"rank(?:ed|ing)?|order(?:ed|ing)?|screen(?:ed|ing)?|scan(?:ned|ning)?|"
    r"sweep(?:ing|ed)?|consider(?:ed|ing)?|explor(?:e|ing|ed|ation)|"
    r"evaluat(?:e|ing|ed|ion)|analy[sz](?:e|ing|ed|is)|inspect(?:ed|ing)?|"
    r"select(?:ed|ing|ion)|choos(?:e|ing)|us(?:e|ing|ed)|add(?:ed|ing)?|"
    r"includ(?:e|ing|ed)|introduc(?:e|ing|ed)|model(?:ed|ing)?|"
    r"calculat(?:e|ing|ed|ion)|comput(?:e|ing|ed)|test(?:ed|ing)?|"
    r"assess(?:ed|ing|ment)?|ensemble|space|analysis|generation|optimization|"
    r"search|sampling|states?\b)\b|"
    r"(?:搜索|寻找|定位|生成|构建|创建|优化|放松|采样|枚举|比较|排序|排列|"
    r"筛选|扫描|扫掠|考虑|探索|评估|分析|检查|选择|使用|加入|引入|增加|"
    r"建模|计算|测试|研究|系综|集合|空间|分析|生成|优化|搜索|采样|状态)",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_SCAN_PATTERN = re.compile(
    r"\b(?:scan(?:ning)?|grid\s+scan|parameter\s+sweep|"
    r"enumerat(?:e|ion)|candidate\s+enumeration)\b|"
    r"扫描|参数扫描|网格|枚举|候选(?:数目|数量|枚举)",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_NUMERIC_PATTERNS = (
    re.compile(
        r"\b(?:within|below|above|under|over|less\s+than|greater\s+than|"
        r"at\s+least|at\s+most|between|approximately|approx\.?|about|"
        r"around|roughly|threshold|cutoff|tolerance|margin|range|criterion|"
        r"criteria|acceptance)\b[^\n<>]{0,60}"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:i|j|%|[A-Za-zÅμµ°]+(?:/[A-Za-zÅμµ°]+)?))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\b(?:rmsd|tolerance|threshold|cutoff|deviation|error|difference|"
        r"distance|barrier|energy\s+(?:difference|gap)|frequency(?:\s+value)?|"
        r"imaginary\s+frequency|temperature)\b|温度)[^\n<>]{0,50}?"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:Å|angstroms?|a\.u\.|kcal(?:ories)?\s*/\s*mol|kJ\s*/\s*mol|"
        r"cm\s*[-−]?\s*1|K|kelvin|%)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:scan(?:ning)?|grid|range|candidate_count|max_candidates|"
        r"orientation_trials|candidate enumeration)\b[^\n<>]{0,60}"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:Å|angstroms?|a\.u\.|kcal(?:ories)?\s*/\s*mol|kJ\s*/\s*mol|"
        r"cm\s*[-−]?\s*1|%)\s*(?:,|and)\s*"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:Å|angstroms?|a\.u\.|kcal(?:ories)?\s*/\s*mol|kJ\s*/\s*mol|"
        r"cm\s*[-−]?\s*1|%)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:Å|angstroms?|a\.u\.|kcal(?:ories)?\s*/\s*mol|kJ\s*/\s*mol|"
        r"cm\s*[-−]?\s*1|%)\s*[^\n<>]{0,24}\b"
        r"(?:rmsd|tolerance|threshold|cutoff|deviation|error|difference|"
        r"distance|barrier|energy\s+(?:difference|gap))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:maxpoints|recalc|points?)\b[^\n<>]{0,20}?"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:Å|angstroms?|a\.u\.|kcal(?:ories)?\s*/\s*mol|kJ\s*/\s*mol|"
        r"cm\s*[-−]?\s*1|%)?(?:\s*[/,]\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)){1,}"
        r"\s*(?:Å|angstroms?|a\.u\.|kcal(?:ories)?\s*/\s*mol|kJ\s*/\s*mol|"
        r"cm\s*[-−]?\s*1|%)?",
        re.IGNORECASE,
    ),
)


_CALCULATION_UNIT_HYPOTHESIS_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9])"
)


_CALCULATION_UNIT_HYPOTHESIS_STATE_PATTERN = re.compile(
    r"\b(?:charge|multiplicity)\b[^\d\n<>]{0,20}"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_METHOD_TOKEN_PATTERN = re.compile(
    r"\b(?:B3LYP|PBE0|PBE|M06(?:-2X)?|\u03c9?B97X(?:-D)?|HF|MP2|"
    r"CCSD(?:\(T\))?|AM1|PM3|PM6|GFN(?:0|1|2)[- ]?xTB|xTB|ALPB|"
    r"6-31G(?:\([^)]*\))?|def2-[A-Za-z0-9+()-]+)\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_SOLVENT_TOKEN_PATTERN = re.compile(
    r"\b(?:water|aqueous|ethanol|methanol|acetonitrile|toluene|benzene|"
    r"dichloromethane|methylene chloride|chloroform|hexane|THF|DMSO|"
    r"DMF|acetone|gas[- ]phase|vacuum|solvent[- ]free)\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_EXTRA_OBJECT_PATTERN = re.compile(
    r"\b(?:additional|extra|new|another|unlisted|undeclared|different|other)\s+"
    r"(?:catalyst|counter[- ]ion|counterion|additive|co-catalyst|intermediate)\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_OMISSION_PATTERN = re.compile(
    r"\b(?:skip|skipping|omit|omitting|without|avoid|avoiding|exclude|"
    r"excluding|drop|dropping|remove|removing|not\s+(?:run|perform|do))\s+"
    r"(?:the\s+)?(?:TS|IRC|transition\s+state|intrinsic\s+reaction\s+"
    r"coordinate|frequency|frequencies)\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_CHINESE_SCOPE_PATTERNS = (
    re.compile(
        r"(?:默认|预设|新增|未声明|额外|严格)?(?:阈值|容差|范围|区间|边界)|"
        r"(?:最大力|最大梯度|RMS\s*力|均方根力|梯度范数|力范数)\s*"
        r"(?:[<≤>=]|低于|高于|小于|大于|不超过|不低于|至少|至多)",
        re.IGNORECASE,
    ),
    re.compile(
        r"热化学|热力学|玻尔兹曼|Boltzmann|吉布斯(?:自由能)?|自由能|"
        r"零点能|热修正",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:额外|新增|未声明|另加|其他)(?:的)?"
        r"(?:对象|催化剂|反应物|产物|中间体|对离子|助剂|溶剂|温度|方法|条件)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:外部|实验|文献|高层)(?:基准|基线|数据|结果|对照)|"
        r"(?:竞争|替代|平行)(?:路径|反应|机理)|"
        r"(?:选择性|对映|立体选择性|\bee\b|ΔΔG)",
        re.IGNORECASE,
    ),
)


_CALCULATION_UNIT_HYPOTHESIS_CHINESE_STATE_PATTERN = re.compile(
    r"(?P<field>总?电荷|多重度|自旋(?:多重度|态))\s*"
    r"(?:为|是|=|:)?\s*(?P<number>[-+]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)


# ==================== 科学假设投影 ====================


def _ordered_unique_source_query_angles(*values: Any) -> list[str]:
    """合并候选来源 Query，保留父项顺序并按完整文本去重。

    来源 Query 是科学上下文而不是运行身份。输入必须是字符串数组（允许空值
    仅用于兼容没有来源的历史内存候选）；其他形状直接失败，避免合成候选丢失
    来源后继续进入比较和排名。
    """

    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("source_query_angles 必须是字符串数组")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("source_query_angles 只能包含非空字符串")
            normalized = item.strip()
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
    return ordered

def project_scientific_hypothesis_identity(
    hypothesis: Mapping[str, Any],
) -> Dict[str, Any]:
    """投影候选的两字段科学正文，供比较、合成和跨批重复诊断共用。

    ``core_claim`` 是旧版运行字段，不能再进入活动模型输入。读取旧快照时只
    尝试复用其中的 ``reasoning`` 或 claims 兼容文本，不把旧核心主张重新发布。
    """

    strategy_name = str(hypothesis.get("strategy_name", "")).strip()
    reasoning = hypothesis.get("reasoning")
    if (not isinstance(reasoning, str) or not reasoning.strip()) and "claims" in hypothesis:
        compatibility = derive_hypothesis_compatibility_text(hypothesis["claims"])
        reasoning = compatibility["reasoning"]
    if (not isinstance(reasoning, str) or not reasoning.strip()) and isinstance(
        hypothesis.get("core_claim"), str
    ):
        # 仅为读取旧快照提供降级正文；该字段绝不会出现在新模型请求中。
        reasoning = hypothesis["core_claim"]
    return {
        "strategy_name": strategy_name,
        "reasoning": str(reasoning or hypothesis.get("detailed_reasoning", "")).strip(),
    }


# ==================== 假设生成智能体 ====================


class _FinalizedHypothesisReplayRunner:
    """把 Dynamic operation 严格重放值按原 stage 身份提供给确定性算法。"""

    def __init__(self, stages: Sequence[ModelStage], values: Sequence[Any]) -> None:
        """绑定已由 Dynamic runner 复核过的 stage 规格和值，不调用 provider。"""

        if len(stages) != len(values) or not stages:
            raise ValueError("finalized hypothesis replay 的 stage/value 数量不一致")
        self._stages = tuple(stages)
        self._values = tuple(copy.deepcopy(value) for value in values)
        self._reserved: dict[int, ModelStage] = {}
        self._completed: set[int] = set()

    @staticmethod
    def _same_stage(left: ModelStage, right: ModelStage) -> bool:
        """按动态清单可保存的字段比较 stage，避免比较不可序列化的运行对象。"""

        return (
            left.stage == right.stage
            and left.stage_kind == right.stage_kind
            and left.validation_kind == right.validation_kind
            and left.query_index == right.query_index
        )

    def _reserve_stage(self, stage: ModelStage) -> int:
        """按原顺序核对一个 stage，并返回其固定清单位置。"""

        index = len(self._reserved) + 1
        if index > len(self._stages) or not self._same_stage(stage, self._stages[index - 1]):
            raise ValueError("finalized hypothesis replay 的 stage 顺序发生变化")
        self._reserved[index] = stage
        return index

    def reserve_model_stages(self, stages: Sequence[ModelStage]) -> list[int]:
        """预留一组并行比较 stage，但不改变已保存回答的顺序。"""

        return [self._reserve_stage(stage) for stage in stages]

    def run_model_stage(self, stage: ModelStage, stage_input: Mapping[str, Any]) -> Any:
        """返回下一个顺序 stage 的严格重放值。"""

        del stage_input
        return self.run_reserved_model_stage(self._reserve_stage(stage), stage, {})

    def run_reserved_model_stage(
        self,
        stage_index: int,
        stage: ModelStage,
        stage_input: Mapping[str, Any],
    ) -> Any:
        """按清单位置返回已校验值，并拒绝重复或身份漂移。"""

        del stage_input
        reserved = self._reserved.get(stage_index)
        if reserved is None or not self._same_stage(reserved, stage):
            raise ValueError("finalized hypothesis replay 的保留 stage 不匹配")
        if stage_index in self._completed:
            raise ValueError("finalized hypothesis replay 不允许重复读取 stage")
        if stage_index < 1 or stage_index > len(self._values):
            raise ValueError("finalized hypothesis replay 的 stage index 越界")
        self._completed.add(stage_index)
        return copy.deepcopy(self._values[stage_index - 1])

    def assert_complete(self) -> None:
        """确认确定性算法消费了全部 finalized stage。"""

        if len(self._reserved) != len(self._stages) or self._completed != set(self._reserved):
            raise ValueError("finalized hypothesis replay 未完整消费 operation stages")


_REGENERATION_OUTCOME_KEYS = {
    "source_query",
    "hypothesis",
    "terminal_classification",
    "final_judge",
    "cited_evidence",
    "attempted_observables",
    "missing_evidence",
    "tool_capability_gap",
}
_REGENERATION_EVIDENCE_KEYS = {
    "evidence_ref",
    "step_ref",
    "evidence_kind",
    "name",
    "value",
    "unit",
    "conditions",
}
_REGENERATION_HYPOTHESIS_KEYS = {"strategy_name", "reasoning"}
_REGENERATION_GAP_KEYS = {
    "result_type",
    "missing_capabilities",
    "reason",
}
_REGENERATION_TERMINAL_CLASSES = {
    "reliably_refuted",
    "insufficient_evidence",
    "tool_capability_gap",
    "supported",
}


def _require_regeneration_mapping(
    value: Any,
    expected_keys: set[str],
    label: str,
) -> Dict[str, Any]:
    """确认再生成协议对象字段恰好符合注册集合。"""

    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{label} 字段不完整")
    return copy.deepcopy(dict(value))


def _validate_regeneration_scientific_context(value: Any) -> Dict[str, Any]:
    """校验再生成使用的科学问题、检索快照和化学条件。"""

    label = "regeneration_package.scientific_context"
    context = _require_regeneration_mapping(
        value,
        _REGENERATION_CONTEXT_KEYS,
        label,
    )
    angles = _validate_regeneration_scientific_context_shape(context, label)
    _validate_regeneration_scientific_context_projection(context, label, angles)
    return context


def _validate_regeneration_scientific_context_shape(
    context: Mapping[str, Any],
    label: str,
) -> list[str]:
    """校验再生成科学上下文中的问题、query 和两个 context 对象。"""

    _validate_regeneration_scientific_question(context["scientific_question"], label)
    angles = _validate_regeneration_query_angles(
        context["initial_query_angles"],
        label,
    )
    _validate_regeneration_context_objects(context, label)
    return angles


def _validate_regeneration_scientific_question(
    value: Any,
    label: str,
) -> None:
    """校验再生成科学问题是非空文本。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 非法")


def _validate_regeneration_query_angles(value: Any, label: str) -> list[str]:
    """校验再生成 query angles 是非空且不重复的字符串数组。"""

    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} 非法")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} 非法")
    return value


def _validate_regeneration_context_objects(
    context: Mapping[str, Any],
    label: str,
) -> None:
    """校验再生成 Retrieval 和 chemistry context 都是 object。"""

    if not isinstance(context["initial_retrieval_phase"], Mapping):
        raise ValueError(f"{label} 非法")
    if not isinstance(context["chemistry_context"], Mapping):
        raise ValueError(f"{label} 非法")


def _validate_regeneration_scientific_context_projection(
    context: Mapping[str, Any],
    label: str,
    angles: Sequence[str],
) -> None:
    """核对再生成科学上下文只能由公共字段投影产生。"""

    try:
        projected = project_hypothesis_regeneration_context(
            scientific_question=context["scientific_question"],
            initial_query_angles=angles,
            initial_retrieval_phase=context["initial_retrieval_phase"],
            chemistry_context=context["chemistry_context"],
        )
    except Exception as exc:
        raise ValueError(
            f"{label} 无法形成固定公共字段视图"
        ) from exc
    if projected != context:
        raise ValueError(f"{label} 包含未注册嵌套字段")


def _validate_regeneration_memory_identity(
    package: MutableMapping[str, Any],
    *,
    expected_task_id: str | None,
    expected_batch_id: str | None,
) -> Mapping[str, Any] | None:
    """校验可选 Working Memory 与当前 task、batch 身份绑定。"""

    if "investigation_working_memory" not in package:
        return None
    if expected_task_id is None or expected_batch_id is None:
        raise ValueError(
            "investigation_working_memory 必须由 bridge 提供 task/batch 身份"
        )
    memory = validate_investigation_working_memory_payload(
        package["investigation_working_memory"],
        label="regeneration_package.investigation_working_memory",
    )
    if memory["task_id"] != expected_task_id:
        raise ValueError(
            "investigation_working_memory.task_id 与当前 operation 不一致"
        )
    if memory["batch_id"] != expected_batch_id:
        raise ValueError(
            "investigation_working_memory.batch_id 与当前 operation 不一致"
        )
    package["investigation_working_memory"] = memory
    return memory


def _validate_regeneration_memory_context_binding(
    memory: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    """核对 Working Memory 的科学上下文与冻结再生成上下文一致。"""

    if memory["scientific_question"] != context["scientific_question"]:
        raise ValueError(
            "investigation_working_memory.scientific_question 与冻结上下文不一致"
        )
    if memory["chemistry_context"] != context["chemistry_context"]:
        raise ValueError(
            "investigation_working_memory.chemistry_context 与冻结上下文不一致"
        )
    try:
        projected = project_hypothesis_regeneration_context(
            scientific_question=memory["scientific_question"],
            initial_query_angles=context["initial_query_angles"],
            initial_retrieval_phase=memory["retrieval_context"],
            chemistry_context=memory["chemistry_context"],
        )["initial_retrieval_phase"]
    except Exception as exc:
        raise ValueError(
            "investigation_working_memory.retrieval_context 无法形成冻结检索视图"
        ) from exc
    if projected != context["initial_retrieval_phase"]:
        raise ValueError(
            "investigation_working_memory.retrieval_context 与冻结上下文不一致"
        )


def _validate_regeneration_evidence_identity(
    evidence: Mapping[str, Any],
    *,
    path: str,
    seen_refs: set[str],
) -> None:
    """校验引用证据编号和可选步骤编号。"""

    evidence_ref = evidence["evidence_ref"]
    if (
        not isinstance(evidence_ref, str)
        or not re.fullmatch(r"e[1-9][0-9]*", evidence_ref)
        or evidence_ref in seen_refs
    ):
        raise ValueError(f"{path} evidence_ref 非法")
    seen_refs.add(evidence_ref)
    step_ref = evidence["step_ref"]
    if step_ref is not None and (
        not isinstance(step_ref, str)
        or not re.fullmatch(r"s[1-9][0-9]*", step_ref)
    ):
        raise ValueError(f"{path} step_ref 非法")


def _validate_regeneration_evidence_details(
    evidence: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """校验引用证据的观测名称、单位和条件对象。"""

    if any(
        not isinstance(evidence[field], str) or not evidence[field].strip()
        for field in ("evidence_kind", "name")
    ):
        raise ValueError(f"{path} 名称非法")
    if evidence["unit"] is not None and not isinstance(evidence["unit"], str):
        raise ValueError(f"{path} unit 非法")
    if not isinstance(evidence["conditions"], Mapping):
        raise ValueError(f"{path} conditions 非法")


def _validate_regeneration_evidence_item(
    value: Any,
    *,
    outcome_index: int,
    evidence_index: int,
    seen_refs: set[str],
) -> None:
    """校验一条引用证据的身份、观测名称和条件对象。"""

    path = (
        f"prior_hypothesis_outcomes[{outcome_index}]."
        f"cited_evidence[{evidence_index}]"
    )
    evidence = _require_regeneration_mapping(value, _REGENERATION_EVIDENCE_KEYS, path)
    _validate_regeneration_evidence_identity(
        evidence,
        path=path,
        seen_refs=seen_refs,
    )
    _validate_regeneration_evidence_details(evidence, path=path)


def _validate_regeneration_evidence_list(value: Any, outcome_index: int) -> None:
    """校验历史结论中的引用证据数组及其唯一编号。"""

    label = f"prior_hypothesis_outcomes[{outcome_index}].cited_evidence"
    if not isinstance(value, list):
        raise ValueError(f"{label} 必须是 array")
    seen_refs: set[str] = set()
    for evidence_index, evidence in enumerate(value):
        _validate_regeneration_evidence_item(
            evidence,
            outcome_index=outcome_index,
            evidence_index=evidence_index,
            seen_refs=seen_refs,
        )


def _validate_regeneration_outcome_header(
    value: Any,
    *,
    outcome_index: int,
    context: Mapping[str, Any],
    has_investigation_memory: bool,
) -> Dict[str, Any]:
    """校验历史假设结论的正文、来源角度、终局分类和裁判理由。"""

    label = f"prior_hypothesis_outcomes[{outcome_index}]"
    outcome = _require_regeneration_mapping(value, _REGENERATION_OUTCOME_KEYS, label)
    _validate_regeneration_outcome_source_query(
        outcome["source_query"],
        label=label,
        context=context,
        has_investigation_memory=has_investigation_memory,
    )
    _validate_regeneration_outcome_hypothesis(outcome["hypothesis"], label)
    if outcome["terminal_classification"] not in _REGENERATION_TERMINAL_CLASSES:
        raise ValueError(f"{label} 终局分类非法")
    _validate_regeneration_outcome_judge(outcome["final_judge"], label)
    return outcome


def _validate_regeneration_outcome_source_query(
    source_query: Any,
    *,
    label: str,
    context: Mapping[str, Any],
    has_investigation_memory: bool,
) -> None:
    """校验历史结论来源 query，并在有 Memory 时绑定初始角度。"""

    if not isinstance(source_query, str) or not source_query.strip():
        raise ValueError(f"{label}.source_query 非法")
    if has_investigation_memory and source_query not in context["initial_query_angles"]:
        raise ValueError(f"{label}.source_query 非法")


def _validate_regeneration_outcome_hypothesis(
    hypothesis: Any,
    label: str,
) -> None:
    """校验历史结论中的策略名和推理正文。"""

    hypothesis = _require_regeneration_mapping(
        hypothesis,
        _REGENERATION_HYPOTHESIS_KEYS,
        f"{label}.hypothesis",
    )
    if any(
        not isinstance(hypothesis[field], str) or not hypothesis[field].strip()
        for field in _REGENERATION_HYPOTHESIS_KEYS
    ):
        raise ValueError(f"{label}.hypothesis 正文非法")


def _validate_regeneration_outcome_judge(
    judge: Any,
    label: str,
) -> None:
    """校验历史结论的最终裁判理由。"""

    judge = _require_regeneration_mapping(
        judge,
        {"reason"},
        f"{label}.final_judge",
    )
    if not isinstance(judge["reason"], str):
        raise ValueError(f"{label}.final_judge 内容非法")


def _validate_regeneration_outcome_details(
    outcome: Mapping[str, Any],
    *,
    outcome_index: int,
) -> None:
    """校验历史假设结论的证据、观测列表和工具能力缺口。"""

    label = f"prior_hypothesis_outcomes[{outcome_index}]"
    _validate_regeneration_evidence_list(
        outcome["cited_evidence"],
        outcome_index,
    )
    for field in ("attempted_observables", "missing_evidence"):
        items = outcome[field]
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(f"{label}.{field} 必须是 array")
    gap = outcome["tool_capability_gap"]
    if gap is not None:
        gap_payload = _require_regeneration_mapping(
            gap,
            _REGENERATION_GAP_KEYS,
            f"{label}.tool_capability_gap",
        )
        if (
            gap_payload["result_type"] != "tool_capability_gap"
            or not isinstance(gap_payload["missing_capabilities"], list)
            or not isinstance(gap_payload["reason"], str)
        ):
            raise ValueError(f"{label}.tool_capability_gap 内容非法")


def _validate_regeneration_hypothesis_outcome(
    value: Any,
    *,
    outcome_index: int,
    context: Mapping[str, Any],
    has_investigation_memory: bool,
) -> None:
    """校验一条历史假设结论的完整科学证据范围。"""

    outcome = _validate_regeneration_outcome_header(
        value,
        outcome_index=outcome_index,
        context=context,
        has_investigation_memory=has_investigation_memory,
    )
    _validate_regeneration_outcome_details(
        outcome,
        outcome_index=outcome_index,
    )


def _validate_regeneration_history(
    package: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    memory_enabled: bool,
    has_investigation_memory: bool,
) -> None:
    """校验当前查询和全部历史结论的可见范围。"""

    current_query = package["current_query"]
    if current_query is not None and (
        not isinstance(current_query, str) or not current_query.strip()
    ):
        raise ValueError("regeneration_package.current_query 非法")
    outcomes = package["prior_hypothesis_outcomes"]
    if not isinstance(outcomes, list):
        raise ValueError("prior_hypothesis_outcomes 必须是 array")
    if memory_enabled and not outcomes:
        raise ValueError("Memory On 的 prior_hypothesis_outcomes 不能为空")
    for index, outcome in enumerate(outcomes):
        _validate_regeneration_hypothesis_outcome(
            outcome,
            outcome_index=index,
            context=context,
            has_investigation_memory=has_investigation_memory,
        )


def _validate_regeneration_generation_config(
    value: Any,
    *,
    query_count: int,
) -> Dict[str, Any]:
    """校验再生成数量配置，并把明确 schema 字段归一到内部 top_n。"""

    raw_config = value
    if isinstance(raw_config, Mapping) and set(raw_config) in ({
        "num_queries",
        "num_hypotheses_per_query",
        "top_n_workflows",
    }, {
        "num_queries",
        "num_hypotheses_per_query",
        "top_n_hypotheses",
    }):
        top_n_key = (
            "top_n_workflows"
            if "top_n_workflows" in raw_config
            else "top_n_hypotheses"
        )
        raw_config = {
            "num_queries": raw_config["num_queries"],
            "num_hypotheses_per_query": raw_config["num_hypotheses_per_query"],
            "top_n": raw_config[top_n_key],
        }
    config = _require_regeneration_mapping(
        raw_config,
        {"num_queries", "num_hypotheses_per_query", "top_n"},
        "regeneration_package.generation_config",
    )
    for field, field_value in config.items():
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value < 1
        ):
            raise ValueError(f"generation_config.{field} 必须是正整数")
    if config["num_queries"] != query_count:
        raise ValueError("initial_query_angles 数量与 generation_config 不一致")
    return config


def _validate_regeneration_source_record_ids(
    package: Mapping[str, Any],
) -> None:
    """校验供 Controller 保存因果来源的字符串编号数组。"""

    if "source_record_ids" not in package:
        return
    source_ids = package["source_record_ids"]
    if not isinstance(source_ids, list) or any(
        not isinstance(item, str) or not item for item in source_ids
    ):
        raise ValueError("source_record_ids 必须是字符串数组")


def _validate_regeneration_package_payload(
    value: Any,
    *,
    memory_enabled: bool,
    expected_task_id: str | None,
    expected_batch_id: str | None,
) -> Dict[str, Any]:
    """组合再生成 package 的身份、历史、配置和公共字段校验。"""

    if not isinstance(value, Mapping) or not _STAGE_REGENERATION_PACKAGE_KEYS <= set(
        value
    ):
        raise ValueError("regeneration_package 必须使用固定阶段视图")
    allowed_keys = (
        _STAGE_REGENERATION_PACKAGE_KEYS
        | _STAGE_REGENERATION_OPTIONAL_INTERNAL_KEYS
    )
    if set(value) - allowed_keys:
        raise ValueError("regeneration_package 包含未注册字段")
    package = copy.deepcopy(dict(value))
    memory = _validate_regeneration_memory_identity(
        package,
        expected_task_id=expected_task_id,
        expected_batch_id=expected_batch_id,
    )
    context = _validate_regeneration_scientific_context(package["scientific_context"])
    if memory is not None:
        _validate_regeneration_memory_context_binding(memory, context)
    package["scientific_context"] = context
    _validate_regeneration_history(
        package,
        context=context,
        memory_enabled=memory_enabled,
        has_investigation_memory=memory is not None,
    )
    if (
        isinstance(package["next_batch_index"], bool)
        or not isinstance(package["next_batch_index"], int)
        or package["next_batch_index"] < 2
    ):
        raise ValueError("regeneration_package.next_batch_index 必须至少为 2")
    package["generation_config"] = _validate_regeneration_generation_config(
        package["generation_config"],
        query_count=len(context["initial_query_angles"]),
    )
    _validate_regeneration_source_record_ids(package)
    if not memory_enabled:
        package["prior_hypothesis_outcomes"] = []
        if "source_record_ids" in package:
            package["source_record_ids"] = []
    _validate_public_regeneration_package_json(
        package,
        label="regeneration_package",
    )
    return package


def _collect_calculation_unit_named_values(value: Any, field: str) -> list[Any]:
    """递归收集 calculation_unit 中同名字段的声明值。"""

    if isinstance(value, Mapping):
        values = [value[field]] if field in value else []
        for child in value.values():
            values.extend(_collect_calculation_unit_named_values(child, field))
        return values
    if isinstance(value, list):
        values: list[Any] = []
        for child in value:
            values.extend(_collect_calculation_unit_named_values(child, field))
        return values
    return []


def _build_calculation_unit_scope_declarations(
    calculation_unit: Mapping[str, Any],
) -> Dict[str, bool]:
    """读取计算单元明确声明的 analysis/job 范围。

    输入是冻结的 calculation_unit；动作是只读取名为 analysis、job 等已登记范围字段的值，
    再判断其中是否明确声明热化学或构象/异构分析；副作用为零。返回两个布尔标记。
    未出现明确声明时返回 False，调用方必须继续拒绝对应扩展，不根据 scope 自由文本或模型常识推断。
    """

    declared_values = [
        value
        for field in _CALCULATION_UNIT_SCOPE_DECLARATION_FIELDS
        for value in _collect_calculation_unit_named_values(calculation_unit, field)
    ]
    declared_text = json.dumps(
        declared_values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    # 声明字段常用 snake_case；只替换下划线后再应用词边界正则，避免
    # ``conformer_analysis`` 被误当成未声明范围，同时保留 ``E-Z`` 的构象写法。
    declared_text = declared_text.replace("_", " ")
    return {
        "thermochemistry_is_declared": bool(
            _CALCULATION_UNIT_QUERY_THERMOCHEMISTRY_PATTERN.search(declared_text)
        ),
        "conformer_is_declared": bool(
            _CALCULATION_UNIT_QUERY_CONFORMER_PATTERN.search(declared_text)
        ),
    }


def _collect_calculation_unit_numeric_values(values: Sequence[Any]) -> set[float]:
    """把 calculation_unit 的数字声明统一为可比较的浮点集合。"""

    numbers: set[float] = set()
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numbers.add(float(value))
            continue
        if isinstance(value, str):
            numbers.update(
                float(number)
                for number in _CALCULATION_UNIT_HYPOTHESIS_NUMBER_RE.findall(value)
            )
    return numbers


def _build_calculation_unit_hypothesis_scope_facts(
    calculation_unit: Mapping[str, Any],
) -> Dict[str, Any]:
    """整理候选正文范围校验所需的已声明方法、条件和数字。"""

    named = lambda field: _collect_calculation_unit_named_values(
        calculation_unit,
        field,
    )
    jobs = named("jobs")
    scan_text = " ".join(
        str(job)
        for job_values in jobs
        for job in (job_values if isinstance(job_values, list) else [job_values])
        if isinstance(job, str)
    )
    return {
        "charge": _collect_calculation_unit_numeric_values(named("charge")),
        "multiplicity": _collect_calculation_unit_numeric_values(
            named("multiplicity")
        ),
        "temperature": _collect_calculation_unit_numeric_values(
            named("temperature") + named("temperature_k")
        ),
        "method": named("method"),
        "basis": named("basis"),
        "solvent": named("solvent"),
        "job_numbers": _collect_calculation_unit_numeric_values(
            named("maxpoints") + named("recalc")
        ),
        "scan_is_declared": bool(
            _CALCULATION_UNIT_HYPOTHESIS_SCAN_PATTERN.search(scan_text)
        ),
        **_build_calculation_unit_scope_declarations(calculation_unit),
    }


def _calculation_unit_number_is_declared(
    number: str,
    declared: set[float],
) -> bool:
    """判断候选正文中的数字是否在计算单元声明集合中。"""

    try:
        numeric = float(number)
    except (TypeError, ValueError):
        return False
    return numeric in declared


def _calculation_unit_token_is_declared(
    token: str,
    values: Sequence[Any],
) -> bool:
    """判断候选正文中的方法或溶剂词是否已由计算单元声明。"""

    normalized = re.sub(r"[\s_-]+", "", token.casefold())
    compact_declared = re.sub(
        r"[\s_-]+",
        "",
        " ".join(str(value) for value in values).casefold(),
    )
    return bool(normalized) and normalized in compact_declared


def _resolve_calculation_unit_declared_numbers(
    context: str,
    facts: Mapping[str, Any],
) -> set[float]:
    """按候选数字所在语义字段返回可复用的声明数值。"""

    lowered = context.casefold()
    if re.search(
        r"\b(?:maxpoints|recalc|points?)\b|(?:最大点数|点数|采样点|路径点)",
        lowered,
    ):
        return facts["job_numbers"]
    if re.search(
        r"\b(?:temperature|temp)\b|\b(?:k|kelvin)\b|温度|开尔文",
        lowered,
    ):
        return facts["temperature"]
    if re.search(r"\b(?:charge|multiplicity)\b", lowered):
        state_numbers = set()
        if "charge" in lowered:
            state_numbers.update(facts["charge"])
        if "multiplicity" in lowered:
            state_numbers.update(facts["multiplicity"])
        return state_numbers
    if "frequency" in lowered or "虚频" in lowered or "频率" in lowered:
        return {0.0, 1.0}
    if facts["scan_is_declared"] and _CALCULATION_UNIT_HYPOTHESIS_SCAN_PATTERN.search(
        context
    ):
        return facts["job_numbers"]
    return set()


def _validate_calculation_unit_hypothesis_text_patterns(
    candidate_text: str,
    facts: Mapping[str, Any],
) -> List[str]:
    """收集候选正文中的未声明范围、对象、扫描或省略条件规则。

    输入是单个字段正文和固定单元事实；动作是按登记正则匹配规则码；副作用为零。
    返回去重后的规则码列表；失败时不抛出领域错误，由上层统一添加字段路径并封装。
    """

    violations: List[str] = []
    if (
        _calculation_unit_hypothesis_conformer_scope_is_active(candidate_text)
        and not facts["conformer_is_declared"]
    ):
        _append_scope_violation_rule(violations, "undeclared_conformer_or_isomer")
    for rule, pattern in _CALCULATION_UNIT_HYPOTHESIS_SCOPE_RULES:
        has_active_scope = any(
            not _hypothesis_scope_match_is_explicitly_excluded(
                candidate_text,
                match,
            )
            for match in pattern.finditer(candidate_text)
        )
        if has_active_scope and not (
            rule == "undeclared_thermochemistry"
            and facts["thermochemistry_is_declared"]
        ):
            _append_scope_violation_rule(violations, rule)
    if (
        any(
            not _hypothesis_scope_match_is_explicitly_excluded(
                candidate_text,
                match,
            )
            for match in _CALCULATION_UNIT_HYPOTHESIS_SCAN_PATTERN.finditer(
                candidate_text
            )
        )
        and not facts["scan_is_declared"]
    ):
        _append_scope_violation_rule(violations, "undeclared_scan")
    if any(
        not _hypothesis_scope_match_is_explicitly_excluded(candidate_text, match)
        for match in _CALCULATION_UNIT_HYPOTHESIS_EXTRA_OBJECT_PATTERN.finditer(
            candidate_text
        )
    ):
        _append_scope_violation_rule(violations, "undeclared_extra_object")
    if _CALCULATION_UNIT_HYPOTHESIS_OMISSION_PATTERN.search(candidate_text):
        _append_scope_violation_rule(violations, "omits_required_job")
    return violations


def _append_scope_violation_rule(
    violations: List[str],
    rule: str,
) -> None:
    """按首次出现顺序加入稳定规则码，并去除同一字段的重复命中。

    输入是可变的规则码列表和固定规则码；动作是原地追加一次；副作用仅限于该列表。
    无返回值；规则码不是字符串时由调用方合同保证，函数不负责替代校验或抛出模型错误。
    """

    if rule not in violations:
        violations.append(rule)


def _validate_calculation_unit_hypothesis_state_numbers(
    candidate_text: str,
    facts: Mapping[str, Any],
) -> List[str]:
    """收集候选正文中替换电荷、多重度数字的规则。

    输入是单个字段正文和固定单元的电子态事实；动作是检查声明数字并记录规则码；副作用为零。
    返回去重后的规则码列表；数字格式异常由调用方的基础 schema 校验处理，本函数不回显正文。
    """

    violations: List[str] = []
    for match in _CALCULATION_UNIT_HYPOTHESIS_STATE_PATTERN.finditer(
        candidate_text
    ):
        state_numbers = (
            facts["charge"]
            if "charge" in match.group(0).casefold()
            else facts["multiplicity"]
        )
        if any(
            not _calculation_unit_number_is_declared(number, state_numbers)
            for number in _CALCULATION_UNIT_HYPOTHESIS_NUMBER_RE.findall(
                match.group(0)
            )
        ):
            _append_scope_violation_rule(violations, "undeclared_electronic_state")
    for match in _CALCULATION_UNIT_HYPOTHESIS_CHINESE_STATE_PATTERN.finditer(
        candidate_text
    ):
        state_numbers = (
            facts["charge"]
            if "电荷" in match.group("field")
            else facts["multiplicity"]
        )
        if not _calculation_unit_number_is_declared(
            match.group("number"),
            state_numbers,
        ):
            _append_scope_violation_rule(violations, "undeclared_electronic_state")
    return violations


def _validate_calculation_unit_hypothesis_chinese_scope(
    candidate_text: str,
    facts: Mapping[str, Any],
) -> List[str]:
    """收集中文正文中未声明的对象、温度或外部判据规则。

    输入是单个字段正文和明确的 analysis/job 范围事实；动作是按中文范围模式检查并区分真正否定和“不同”用法；副作用为零。
    返回去重后的规则码列表；模式输入不满足字符串约定时由调用方负责拒绝，不在此处回显正文。
    """

    violations: List[str] = []
    pattern_rules = (
        "undeclared_acceptance_threshold",
        "undeclared_thermochemistry",
        "undeclared_extra_object",
        "undeclared_external_or_competing_evidence",
    )
    for pattern_index, pattern in enumerate(
        _CALCULATION_UNIT_HYPOTHESIS_CHINESE_SCOPE_PATTERNS
    ):
        rule = pattern_rules[pattern_index]
        if (
            rule == "undeclared_thermochemistry"
            and facts["thermochemistry_is_declared"]
        ):
            continue
        for match in pattern.finditer(candidate_text):
            prefix = candidate_text[max(0, match.start() - 12) : match.start()]
            if re.search(
                r"(?:未|不(?!同)|无|没有|不得|禁止|避免|无需)[^，。；:：]{0,8}$",
                prefix,
            ):
                continue
            _append_scope_violation_rule(violations, pattern_rules[pattern_index])
    return violations


def _validate_calculation_unit_hypothesis_tokens(
    candidate_text: str,
    facts: Mapping[str, Any],
) -> List[str]:
    """收集候选正文中未声明的方法、基组或溶剂规则。

    输入是单个字段正文和固定单元的方法、基组、溶剂事实；动作是核对词项是否已声明；副作用为零。
    返回去重后的规则码列表；固定事实缺失等合同错误向调用方传播，不把模型正文放进结果。
    """

    violations: List[str] = []
    declared_method_or_basis = facts["method"] + facts["basis"]
    for match in _CALCULATION_UNIT_HYPOTHESIS_METHOD_TOKEN_PATTERN.finditer(
        candidate_text
    ):
        token = match.group(0)
        allowed_values = declared_method_or_basis
        if token.casefold() == "alpb":
            allowed_values = allowed_values + facts["solvent"]
        if (
            not _calculation_unit_token_is_declared(token, allowed_values)
            and not _hypothesis_scope_match_is_explicitly_excluded(
                candidate_text,
                match,
            )
        ):
            _append_scope_violation_rule(violations, "undeclared_method_or_basis")
    for match in _CALCULATION_UNIT_HYPOTHESIS_SOLVENT_TOKEN_PATTERN.finditer(
        candidate_text
    ):
        if (
            not _calculation_unit_token_is_declared(
                match.group(0),
                facts["solvent"],
            )
            and not _hypothesis_scope_match_is_explicitly_excluded(
                candidate_text,
                match,
            )
        ):
            _append_scope_violation_rule(violations, "undeclared_solvent")
    return violations


def _validate_calculation_unit_hypothesis_numeric_phrases(
    candidate_text: str,
    facts: Mapping[str, Any],
) -> List[str]:
    """收集候选正文中未声明的阈值、范围和数值判据规则。

    输入是单个字段正文和固定单元数值事实；动作是按语义字段核对每个数字；副作用为零。
    返回去重后的规则码列表；解析异常由调用方的 schema 合同处理，函数不保存或回显原文。
    """

    violations: List[str] = []
    for pattern_index, pattern in enumerate(
        _CALCULATION_UNIT_HYPOTHESIS_NUMERIC_PATTERNS
    ):
        for match in pattern.finditer(candidate_text):
            numeric_fragment = _CALCULATION_UNIT_HYPOTHESIS_METHOD_TOKEN_PATTERN.sub(
                " ",
                match.group(0),
            )
            numbers = _CALCULATION_UNIT_HYPOTHESIS_NUMBER_RE.findall(
                numeric_fragment
            )
            context = match.group(0).casefold()
            declared = _resolve_calculation_unit_declared_numbers(context, facts)
            if pattern_index == 6 and not declared:
                clause_start = max(
                    candidate_text.rfind(separator, 0, match.start())
                    for separator in ("。", "；", ";", "\n")
                )
                following_boundaries = [
                    boundary
                    for separator in ("。", "；", ";", "\n")
                    if (boundary := candidate_text.find(separator, match.end()))
                    >= 0
                ]
                clause_end = (
                    min(following_boundaries)
                    if following_boundaries
                    else len(candidate_text)
                )
                declared = _resolve_calculation_unit_declared_numbers(
                    candidate_text[clause_start + 1 : clause_end].casefold(),
                    facts,
                )
            if any(
                not _calculation_unit_number_is_declared(number, declared)
                for number in numbers
            ):
                _append_scope_violation_rule(violations, "undeclared_numeric_criterion")
    return violations


def _format_calculation_unit_hypothesis_scope_error(
    violations: Sequence[Tuple[str, str]],
) -> str:
    """把有限的路径和规则码编码为不含模型正文的领域错误。

    输入是校验器收集的路径/规则二元组；动作是去重并截断到固定上限后编码；副作用为零。
    返回稳定 JSON 诊断字符串；若调用方传入非法形状，序列化异常向调用方传播而不拼接模型正文。
    """

    details = []
    seen = set()
    for path, rule in violations:
        pair = (path, rule)
        if pair in seen:
            continue
        seen.add(pair)
        details.append({"path": path, "rule": rule})
        if len(details) == _CALCULATION_UNIT_HYPOTHESIS_SCOPE_DETAIL_LIMIT:
            break
    return _CALCULATION_UNIT_HYPOTHESIS_SCOPE_DETAIL_PREFIX + json.dumps(
        details,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _scope_match_is_explicitly_negated(text: str, match: re.Match[str]) -> bool:
    """判断范围词前的有限英文语境是否明确排除了该范围。

    输入是完整字段正文和范围正则的一次匹配；动作是只读取匹配词前 48 个字符，复用 Query
    已验证的 ``without/no/not`` locality 规则；副作用为零。返回 True 表示直接排除该范围，
    ``not only``、远处否定或含糊转折不会被当作排除，避免 Hypothesis 与 Query 的边界漂移。
    """

    prefix = text[max(0, match.start() - 48) : match.start()]
    return _CALCULATION_UNIT_QUERY_NEGATED_SCOPE_PATTERN.search(prefix) is not None


_CALCULATION_UNIT_HYPOTHESIS_SCOPE_NEGATION_PATTERN = re.compile(
    r"\b(?:without|no|never|neither|nor|exclude|excluding|omit|omitting|"
    r"avoid|avoiding)\b|"
    r"\b(?:do|does|did|should|must)\s+not\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_SCOPE_EXCLUSION_VERB_PATTERN = re.compile(
    r"\b(?:extend(?:ed|ing)?\s+to|introduc(?:e|ed|ing)|requir(?:e|ed|ing)|"
    r"perform(?:ed|ing)?|conduct(?:ed|ing)?|carry\s+out|add(?:ed|ing)?|"
    r"includ(?:e|ed|ing)|calculat(?:e|ed|ing)|comput(?:e|ed|ing)|"
    r"us(?:e|ed|ing)|compar(?:e|ed|ing)|report(?:ed|ing)?|deriv(?:e|ed|ing)|"
    r"evaluat(?:e|ed|ing)|analy[sz](?:e|ed|ing)|measure(?:d|ing)?|"
    r"apply|appl(?:ied|ying)|obtain(?:ed|ing)?|inspect(?:ed|ing)?|"
    r"interpret(?:ed|ing)?|test(?:ed|ing)?|assess(?:ed|ing)?|"
    r"replac(?:e|ed|ing)|substitut(?:e|ed|ing)|switch(?:ed|ing)?)\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_SCOPE_EXCLUSION_NOUN_PATTERN = re.compile(
    r"\b(?:evidence|proof|indication|indications|sign|signs)\s+of\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_SCOPE_TURN_PATTERN = re.compile(
    r"\b(?:but|however|although|yet|except|unless)\b|[,.;:!?]",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_SCOPE_NEW_ACTION_PATTERN = re.compile(
    r"\b(?:and|or)\s+(?:then\s+)?(?:calculat(?:e|ed|es)|"
    r"comput(?:e|ed|es)|use(?:d|s)?|compar(?:e|ed|es)|"
    r"report(?:ed|s)?|deriv(?:e|ed|es)|evaluat(?:e|ed|es)|"
    r"analy[sz](?:e|ed|es)|rank(?:ed|s)?|measure(?:d|s)?|apply|"
    r"appl(?:ied|ies)|obtain(?:ed|s)?|inspect(?:ed|s)?|"
    r"interpret(?:ed|s)?|test(?:ed|s)?|assess(?:ed|es)?|"
    r"scan(?:ned|s)?|sweep(?:ed|s)?|enumerat(?:e|ed|es)?)\b",
    re.IGNORECASE,
)


_CALCULATION_UNIT_HYPOTHESIS_SCOPE_LIST_CONNECTOR_PATTERN = re.compile(
    r"\b(?:and|or)\b(?:\s+(?:the|a|an|any|additional|further|new|extra|"
    r"another|other))?\s*$",
    re.IGNORECASE,
)


# 冻结真实 Case 1 表述的最长否定前缀为 5 个英文词；保留有限余量但不接受无限回溯。
_CALCULATION_UNIT_HYPOTHESIS_SCOPE_MAX_NEGATION_TOKENS = 12


def _hypothesis_scope_match_is_explicitly_excluded(
    text: str,
    match: re.Match[str],
) -> bool:
    """判断 Hypothesis 范围词或未声明 token 是否属于同一英文否定从句的排除内容。

    输入是完整候选字段和一个 scope/token 匹配；动作是定位最近的否定词，检查其到目标词之间是否
    仍在同一从句，并确认目标词是排除谓词的直接宾语、同一范围名词列表的一项，或 ``evidence of``
    等明确排除名词短语的对象；副作用为零。
    逗号、转折词或新的肯定动作会终止否定范围，``not only`` 也不会被当作排除。返回 True 表示
    该范围词只是边界说明，可以跳过未声明范围规则；实际计算或跨从句否定返回 False。此 helper
    只服务 Hypothesis，Query 继续使用原有的短 locality helper。
    """

    prefix = text[: match.start()]
    marker_matches = list(
        _CALCULATION_UNIT_HYPOTHESIS_SCOPE_NEGATION_PATTERN.finditer(prefix)
    )
    if not marker_matches:
        return False
    marker = marker_matches[-1]
    segment = prefix[marker.end() :]
    if _CALCULATION_UNIT_HYPOTHESIS_SCOPE_TURN_PATTERN.search(segment):
        return False
    if (
        len(re.findall(r"[A-Za-z][A-Za-z'/-]*", segment))
        > _CALCULATION_UNIT_HYPOTHESIS_SCOPE_MAX_NEGATION_TOKENS
    ):
        return False
    if re.match(r"\s+(?:only|just|merely)\b", segment, re.IGNORECASE):
        return False
    if re.fullmatch(r"(?:\s+[A-Za-z][A-Za-z'/-]*){0,2}\s*", segment):
        return True
    if _CALCULATION_UNIT_HYPOTHESIS_SCOPE_NEW_ACTION_PATTERN.search(segment):
        return False
    if _CALCULATION_UNIT_HYPOTHESIS_SCOPE_EXCLUSION_VERB_PATTERN.search(segment):
        return True
    if _CALCULATION_UNIT_HYPOTHESIS_SCOPE_EXCLUSION_NOUN_PATTERN.search(segment):
        return True
    prior_scope_match = any(
        pattern.search(segment)
        for pattern in (
            _CALCULATION_UNIT_HYPOTHESIS_THERMOCHEMISTRY_SCOPE_PATTERN,
            _CALCULATION_UNIT_HYPOTHESIS_CONFORMER_TOKEN_PATTERN,
        )
    )
    return prior_scope_match and (
        _CALCULATION_UNIT_HYPOTHESIS_SCOPE_LIST_CONNECTOR_PATTERN.search(segment)
        is not None
    )


def _calculation_unit_hypothesis_conformer_scope_is_active(text: str) -> bool:
    """判断候选是否以动作方式新增未声明的构象或异构范围。

    输入是一个 strategy_name 或 reasoning 字段；动作是逐个检查范围词的局部片段：固定作业的
    endpoint/IRC/虚频结果、不同构象的反例观测和“构象调整”不触发拒绝；搜索、生成、优化、
    采样、枚举、比较、排序、扫描、考虑、分析、构象系综以及额外/替代构象仍触发拒绝。副作用为零。
    返回 True 表示应记录 ``undeclared_conformer_or_isomer``。
    """

    for match in _CALCULATION_UNIT_HYPOTHESIS_CONFORMER_TOKEN_PATTERN.finditer(text):
        if _hypothesis_scope_match_is_explicitly_excluded(text, match) or re.search(
            r"(?:未|不(?!同)|无|没有|不得|禁止|避免|无需)"
            r"[^，。；:：]{0,8}$",
            text[max(0, match.start() - 12) : match.start()],
        ):
            continue
        if re.search(r"ensemble|系综|集合|采样", match.group(0), re.IGNORECASE):
            return True
        separators = "。！？!?；;.\n"
        clause_start = max(
            text.rfind(separator, 0, match.start()) for separator in separators
        )
        following = [
            boundary
            for separator in separators
            if (boundary := text.find(separator, match.end())) >= 0
        ]
        clause_end = min(following) if following else len(text)
        prefix = text[clause_start + 1 : match.start()]
        suffix = text[match.end() : clause_end]
        english_prefix = " ".join(
            re.findall(r"[A-Za-z][A-Za-z'/-]*", prefix)[-4:]
        )
        english_suffix = " ".join(
            re.findall(r"[A-Za-z][A-Za-z'/-]*", suffix)[:4]
        )
        chinese_window = prefix[-10:] + suffix[:12]
        if (
            _CALCULATION_UNIT_HYPOTHESIS_CONFORMER_ACTION_PATTERN.search(
                english_prefix
            )
            or _CALCULATION_UNIT_HYPOTHESIS_CONFORMER_ACTION_PATTERN.search(
                english_suffix
            )
            or _CALCULATION_UNIT_HYPOTHESIS_CONFORMER_ACTION_PATTERN.search(
                chinese_window
            )
        ):
            return True
    return False


def _collect_calculation_unit_query_scope_violations(
    queries: Sequence[str],
    declared_text: str,
    scope_declarations: Mapping[str, bool],
) -> List[Tuple[str, str]]:
    """按 Query 顺序收集有限的数组路径和稳定范围规则。

    输入是已通过基础格式校验的 Query、固定单元的序列化声明及其范围事实；动作是复用现有范围正则，
    只记录 ``/index`` 与登记规则码，最多保留固定上限；副作用为零。返回值不含 Query 原文，
    结构化错误由调用方统一封装。
    """

    violations: List[Tuple[str, str]] = []

    def record(index: int, rule: str) -> None:
        pair = (f"/{index}", rule)
        if pair not in violations and len(violations) < _CALCULATION_UNIT_HYPOTHESIS_SCOPE_DETAIL_LIMIT:
            violations.append(pair)

    for index, query in enumerate(queries):
        if len(violations) >= _CALCULATION_UNIT_HYPOTHESIS_SCOPE_DETAIL_LIMIT:
            break
        for pattern in _CALCULATION_UNIT_QUERY_NUMERIC_PATTERNS:
            for match in pattern.finditer(query):
                if match.group(0).strip() not in declared_text:
                    record(index, "undeclared_numeric_criterion")
                    break
            if len(violations) >= _CALCULATION_UNIT_HYPOTHESIS_SCOPE_DETAIL_LIMIT:
                break
        if len(violations) >= _CALCULATION_UNIT_HYPOTHESIS_SCOPE_DETAIL_LIMIT:
            break

        criterion_match = _CALCULATION_UNIT_QUERY_UNDECLARED_CRITERION_PATTERN.search(
            query
        )
        if (
            criterion_match is not None
            and criterion_match.group(0).casefold() not in declared_text.casefold()
        ):
            record(index, "undeclared_acceptance_threshold")
        if _CALCULATION_UNIT_QUERY_EXTERNAL_PATTERN.search(query):
            record(index, "undeclared_external_or_competing_evidence")
        if _CALCULATION_UNIT_QUERY_COMPETITION_PATTERN.search(query):
            record(index, "undeclared_external_or_competing_evidence")
        if _CALCULATION_UNIT_QUERY_INITIAL_GUESS_METHOD_PATTERN.search(query):
            record(index, "undeclared_initial_guess_method")
        if (
            not scope_declarations["thermochemistry_is_declared"]
            and any(
                not _scope_match_is_explicitly_negated(query, match)
                for match in _CALCULATION_UNIT_QUERY_THERMOCHEMISTRY_PATTERN.finditer(
                    query
                )
            )
        ):
            record(index, "undeclared_thermochemistry")
        if (
            any(
                not _scope_match_is_explicitly_negated(query, match)
                for match in _CALCULATION_UNIT_QUERY_CONFORMER_PATTERN.finditer(
                    query
                )
            )
            and not scope_declarations["conformer_is_declared"]
        ):
            record(index, "undeclared_conformer_or_isomer")
    return violations


def _collect_calculation_unit_hypothesis_scope_violations(
    hypothesis: Mapping[str, Any],
    facts: Mapping[str, Any],
    item_path: str,
) -> List[Tuple[str, str]]:
    """按字段和固定校验顺序收集一项候选的范围违规。

    输入是一个候选、固定单元事实和候选数组路径；动作是逐字段运行既有规则；副作用为零。
    返回带字段路径的规则码列表；规则本身不抛错，结构化错误由批次调用方统一封装。
    """

    violations: List[Tuple[str, str]] = []
    for field in ("strategy_name", "reasoning"):
        candidate_text = str(hypothesis.get(field, ""))
        rules = []
        rules.extend(
            _validate_calculation_unit_hypothesis_text_patterns(candidate_text, facts)
        )
        rules.extend(
            _validate_calculation_unit_hypothesis_state_numbers(candidate_text, facts)
        )
        rules.extend(
            _validate_calculation_unit_hypothesis_chinese_scope(candidate_text, facts)
        )
        rules.extend(
            _validate_calculation_unit_hypothesis_tokens(candidate_text, facts)
        )
        rules.extend(
            _validate_calculation_unit_hypothesis_numeric_phrases(
                candidate_text,
                facts,
            )
        )
        for rule in rules:
            pair = (f"{item_path}/{field}", rule)
            if pair not in violations:
                violations.append(pair)
    return violations


class HypothesisAgent:
    """执行假设生成、关系判断、归并和排名。"""

    def __init__(
        self,
        comparison_workers: int | str = 4,
        model_provider: Optional[
            Callable[[List[Dict[str, str]]], Any]
        ] = None,
        *,
        hypothesis_generation_workers: int = 1,
    ) -> None:
        """设置比较与假设生成并发数，并允许测试注入模型响应入口。

        ``hypothesis_generation_workers`` 只控制各 Query 的 Hypothesis 或
        regeneration 子调用；它与关系判断、合成和排名使用的
        ``comparison_workers`` 独立。通用 Agent 默认串行，固定四 Query 的
        复现入口另行显式设置为 4。
        """

        self.model_provider = model_provider
        self._operation_runner: Optional[DynamicModelOperationRunner] = None
        if isinstance(comparison_workers, bool):
            raise ValueError("comparison_workers 不能是布尔值")
        if isinstance(comparison_workers, str):
            if comparison_workers.casefold() != "auto":
                raise ValueError("comparison_workers 只能是正整数、0 或 auto")
            self.comparison_workers = 0
        elif isinstance(comparison_workers, int):
            if comparison_workers < 0:
                raise ValueError("comparison_workers 不能是负数")
            self.comparison_workers = comparison_workers
        else:
            raise ValueError("comparison_workers 必须是正整数、0 或 auto")
        try:
            self.hypothesis_generation_workers = max(
                1,
                int(hypothesis_generation_workers),
            )
        except (TypeError, ValueError):
            logger.warning(
                "无效的假设生成并发数 %r，回退为串行执行",
                hypothesis_generation_workers,
            )
            self.hypothesis_generation_workers = 1

    # ==================== 模型调用与协议校验 ====================

    def _call_model_provider(self, request: Mapping[str, Any]) -> Any:
        """调用模型并保留完整响应信息。

        完整响应里包含正文是否被截断等信息；只取正文会丢失这些判断依据。
        测试时可以注入本地 provider，生产环境则调用已配置的通用模型。
        """

        if self.model_provider is not None:
            return self.model_provider(copy.deepcopy(request["messages"]))
        return call_compiled_model_request(request)

    def _call_model_stage(
        self,
        *,
        stage_name: str,
        base_input: Mapping[str, Any],
        specification: ModelStage,
        reserved_stage_index: Optional[int] = None,
    ) -> Any:
        """通过当前 hypothesis operation runner 完成一次严格模型调用。

        operation runner 负责请求重试、响应校验和清单记录；没有活动 runner 时直接
        抛错，避免绕过正式 operation 产生无法审计的模型调用。
        """

        operation_runner = self._operation_runner
        if operation_runner is not None:
            try:
                if reserved_stage_index is None:
                    return operation_runner.run_model_stage(
                        specification,
                        base_input,
                    )
                return operation_runner.run_reserved_model_stage(
                    reserved_stage_index,
                    specification,
                    base_input,
                )
            except ModelOperationStageError as exc:
                raise HypothesisProtocolError(
                    stage_name,
                    exc.failure_kind,
                    query_index=specification.query_index,
                ) from exc
        raise RuntimeError("hypothesis model stage 缺少活动 operation runner")

    @staticmethod
    def _validate_stage_tool_projection_shape(
        raw_item: Any,
        *,
        index: int,
        required_fields: set[str],
        optional_fields: set[str],
    ) -> Mapping[str, Any]:
        """校验一个 Action 投影的字段白名单。"""

        if not isinstance(raw_item, Mapping):
            raise ValueError(
                f"stage_tool_projection[{index}] MiniChem 字段集合不正确"
            )
        item_fields = set(raw_item)
        if not required_fields.issubset(item_fields) or not item_fields.issubset(
            required_fields | optional_fields
        ):
            raise ValueError(
                f"stage_tool_projection[{index}] MiniChem 字段集合不正确"
            )
        return raw_item

    @staticmethod
    def _validate_stage_tool_projection_action_identity(
        item: Mapping[str, Any],
        *,
        action_ids: set[str],
    ) -> None:
        """校验 Action 编号、描述和唯一性。"""

        action_id = item["action_id"]
        if (
            not isinstance(action_id, str)
            or not action_id.strip()
            or action_id in action_ids
        ):
            raise ValueError("MiniChem stage_tool_projection action_id 非法或重复")
        action_ids.add(action_id)
        description = item["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError("MiniChem stage_tool_projection description 非法")

    @staticmethod
    def _validate_stage_tool_projection_backend_facts(
        item: Mapping[str, Any],
    ) -> list[str]:
        """校验 Action 声明的后端编号和后端对象。"""

        backend_ids = item["backend_ids"]
        backends = item["backends"]
        HypothesisAgent._validate_stage_tool_projection_backend_ids(backend_ids)
        HypothesisAgent._validate_stage_tool_projection_backend_objects(backends)
        return backend_ids

    @staticmethod
    def _validate_stage_tool_projection_backend_ids(value: Any) -> None:
        """校验 Action 后端编号是非空字符串数组。"""

        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ValueError("MiniChem stage_tool_projection backend 事实非法")

    @staticmethod
    def _validate_stage_tool_projection_backend_objects(value: Any) -> None:
        """校验 Action 后端对象是数组。"""

        if not isinstance(value, list) or any(
            not isinstance(item, Mapping) for item in value
        ):
            raise ValueError("MiniChem stage_tool_projection backend 事实非法")

    @staticmethod
    def _validate_stage_tool_projection_identity(
        item: Mapping[str, Any],
        *,
        action_ids: set[str],
    ) -> list[str]:
        """校验 Action 身份、描述和后端事实。"""

        HypothesisAgent._validate_stage_tool_projection_action_identity(
            item,
            action_ids=action_ids,
        )
        return HypothesisAgent._validate_stage_tool_projection_backend_facts(item)

    @staticmethod
    def _validate_stage_tool_projection_data_fields(
        item: Mapping[str, Any],
        *,
        index: int,
    ) -> None:
        """校验 Action 输入、输出、方法和数据动作字段。"""

        for field in ("required_inputs", "optional_inputs", "output_semantic_types"):
            if not isinstance(item[field], list):
                raise ValueError(
                    f"stage_tool_projection.{field} 必须是数组"
                )
        if not isinstance(item["method_spec"], Mapping):
            raise ValueError("MiniChem stage_tool_projection.method_spec 必须是 object")
        if not isinstance(item["data_action"], bool):
            raise ValueError("MiniChem stage_tool_projection.data_action 必须是 bool")

    @staticmethod
    def _validate_stage_tool_projection_companion_contracts(
        item: Mapping[str, Any],
        *,
        backend_ids: Sequence[str],
        index: int,
    ) -> None:
        """校验 Action 可选的后端合同和 Planner 约束。"""

        if "backend_request_contracts" in item:
            HypothesisAgent._validate_backend_request_contracts(
                item["backend_request_contracts"],
                backend_ids,
                index,
            )
        if "planner_constraints" in item:
            HypothesisAgent._validate_planner_constraints(
                item["planner_constraints"],
                backend_ids,
                index,
            )

    @staticmethod
    def _validate_stage_tool_projection_item(
        raw_item: Any,
        *,
        index: int,
        action_ids: set[str],
        required_fields: set[str],
        optional_fields: set[str],
    ) -> Dict[str, Any]:
        """校验一个 MiniChem Action 投影及其后端合同。"""

        item = HypothesisAgent._validate_stage_tool_projection_shape(
            raw_item,
            index=index,
            required_fields=required_fields,
            optional_fields=optional_fields,
        )
        backend_ids = HypothesisAgent._validate_stage_tool_projection_identity(
            item,
            action_ids=action_ids,
        )
        HypothesisAgent._validate_stage_tool_projection_data_fields(
            item,
            index=index,
        )
        HypothesisAgent._validate_stage_tool_projection_companion_contracts(
            item,
            backend_ids=backend_ids,
            index=index,
        )
        return copy.deepcopy(dict(item))

    @staticmethod
    def _validate_stage_tool_projection(value: Any) -> List[Dict[str, Any]]:
        """复核 MiniChem Action/Backend 投影，并拒绝夹带执行内部元数据。"""

        if not isinstance(value, list) or not value:
            raise ValueError("MiniChem stage_tool_projection 必须是非空数组")
        required_fields = {
            "action_id",
            "action_version",
            "description",
            "backend_ids",
            "backends",
            "required_inputs",
            "optional_inputs",
            "method_spec",
            "primary_output",
            "output_semantic_types",
            "selection_policy",
            "data_action",
        }
        optional_fields = {"backend_request_contracts", "planner_constraints"}
        action_ids: set[str] = set()
        return [
            HypothesisAgent._validate_stage_tool_projection_item(
                raw_item,
                index=index,
                action_ids=action_ids,
                required_fields=required_fields,
                optional_fields=optional_fields,
            )
            for index, raw_item in enumerate(value)
        ]

    @staticmethod
    def _validate_planner_constraints(
        value: Any,
        backend_ids: Sequence[str],
        action_index: int,
    ) -> None:
        """校验模型可见的当前任务 Action/Backend 不可用事实。

        输入必须只含非空 ``unavailable_backend_ids`` 和非空原因；动作核对每个
        后端已经由当前 Action 声明且没有重复，不修改投影。字段、类型、身份或
        原因不合法时抛出 ``ValueError``，不会调用模型或工具。
        """

        validated = HypothesisAgent._validate_planner_constraints_shape(
            value,
            action_index,
        )
        unavailable = validated["unavailable_backend_ids"]
        HypothesisAgent._validate_planner_constraints_backend_ids(
            unavailable,
            backend_ids,
            action_index,
        )
        reason = validated["unavailable_reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "MiniChem stage_tool_projection"
                f"[{action_index}].planner_constraints.unavailable_reason 必须是非空字符串"
            )

    @staticmethod
    def _validate_planner_constraints_shape(
        value: Any,
        action_index: int,
    ) -> Mapping[str, Any]:
        """校验 Planner 约束对象只含两个公开字段。"""

        expected_fields = {"unavailable_backend_ids", "unavailable_reason"}
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise ValueError(
                "MiniChem stage_tool_projection"
                f"[{action_index}].planner_constraints 字段集合不正确"
            )
        return value

    @staticmethod
    def _validate_planner_constraints_backend_ids(
        unavailable: Any,
        backend_ids: Sequence[str],
        action_index: int,
    ) -> None:
        """校验 Planner 标出的不可用后端均已由 Action 声明。"""

        if (
            not isinstance(unavailable, list)
            or not unavailable
            or any(not isinstance(item, str) or not item.strip() for item in unavailable)
            or len(set(unavailable)) != len(unavailable)
        ):
            raise ValueError(
                "MiniChem stage_tool_projection"
                f"[{action_index}].planner_constraints.unavailable_backend_ids 非法"
            )
        if any(item not in set(backend_ids) for item in unavailable):
            raise ValueError(
                "MiniChem stage_tool_projection"
                f"[{action_index}].planner_constraints.unavailable_backend_ids 非法"
            )

    @staticmethod
    def _validate_backend_request_contracts(
        value: Any,
        backend_ids: Sequence[str],
        action_index: int,
    ) -> None:
        """校验 Action 下的后端专属请求合同，拒绝重复或未声明后端。

        该字段只在冻结目录包含 companion 合同时出现。每条合同必须完整描述
        一个已声明的 backend，并把请求合同和输出合同保存为对象；其他元数据不
        会被模型投影接受。
        """

        if not isinstance(value, list) or not value:
            raise ValueError(
                "MiniChem stage_tool_projection"
                f"[{action_index}].backend_request_contracts 必须是非空数组"
            )
        declared_backends = set(backend_ids)
        seen_backends = set()
        expected_fields = {"backend_id", "request_contract", "output_contract"}
        for contract_index, raw_contract in enumerate(value):
            if (
                not isinstance(raw_contract, Mapping)
                or set(raw_contract) != expected_fields
            ):
                raise ValueError(
                    "MiniChem stage_tool_projection"
                    f"[{action_index}].backend_request_contracts"
                    f"[{contract_index}] 字段集合不正确"
                )
            backend_id = raw_contract["backend_id"]
            if not isinstance(backend_id, str) or not backend_id.strip():
                raise ValueError(
                    "MiniChem stage_tool_projection.backend_request_contracts"
                    " backend_id 非法"
                )
            if backend_id not in declared_backends:
                raise ValueError(
                    "MiniChem stage_tool_projection.backend_request_contracts"
                    f" backend_id 未由 Action 声明：{backend_id}"
                )
            if backend_id in seen_backends:
                raise ValueError(
                    "MiniChem stage_tool_projection.backend_request_contracts"
                    f" backend_id 重复：{backend_id}"
                )
            seen_backends.add(backend_id)
            if not isinstance(raw_contract["request_contract"], Mapping):
                raise ValueError(
                    "MiniChem stage_tool_projection.backend_request_contracts"
                    " request_contract 必须是 object"
                )
            if not isinstance(raw_contract["output_contract"], Mapping):
                raise ValueError(
                    "MiniChem stage_tool_projection.backend_request_contracts"
                    " output_contract 必须是 object"
                )

    @staticmethod
    def _normalize_generated_hypotheses(
        value: Any,
        machine_tool_identifiers: Sequence[str] = (),
        *,
        calculation_unit: Mapping[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """校验模型返回的两字段科学正文，并排除机械重复候选。

        ``machine_tool_identifiers`` 保留在函数签名中是为了兼容旧调用方，但不再
        被用于假设校验；工具能力只能由 Planner 和 MiniChem 合同判断。存在非空
        ``calculation_unit`` 时，候选正文还必须通过固定范围校验。
        """

        del machine_tool_identifiers
        normalized = []
        identities = set()
        for item in value:
            if not isinstance(item, Mapping) or set(item) != {
                "strategy_name",
                "reasoning",
            }:
                raise ValueError(
                    "模型 hypothesis 必须且只含 strategy_name 与 reasoning"
                )
            fields = {}
            limits = {"strategy_name": 600, "reasoning": 3200}
            for field, limit in limits.items():
                raw_text = item[field]
                if not isinstance(raw_text, str) or not raw_text.strip():
                    raise ValueError(f"hypothesis.{field} 必须是非空字符串")
                text = raw_text.strip()
                if limit is not None and len(text) > limit:
                    raise ValueError(f"hypothesis.{field} 超过 {limit} 字符")
                fields[field] = text
            identity = (
                fields["strategy_name"],
                fields["reasoning"],
            )
            if identity in identities:
                raise ValueError("同一 hypothesis subcall 不得包含完全重复项")
            identities.add(identity)
            normalized.append(fields)
        HypothesisAgent._validate_calculation_unit_hypothesis_scope(
            normalized,
            calculation_unit,
        )
        return normalized

    @staticmethod
    def _validate_calculation_unit_hypothesis_scope(
        hypotheses: Sequence[Mapping[str, Any]],
        calculation_unit: Mapping[str, Any] | None,
    ) -> None:
        """校验候选是否仍只描述给定计算单元。

        输入是候选数组和冻结 calculation_unit；动作是收集有限的 item/字段规则并抛出安全摘要；
        副作用为零。返回值为无；通过时不抛出，越界时只抛出稳定路径/规则码，不回显候选正文。
        """

        if not isinstance(calculation_unit, Mapping) or not calculation_unit:
            return
        facts = _build_calculation_unit_hypothesis_scope_facts(calculation_unit)
        violations: List[Tuple[str, str]] = []
        for index, hypothesis in enumerate(hypotheses):
            item_path = f"/{index}"
            if not isinstance(hypothesis, Mapping):
                continue
            violations.extend(
                _collect_calculation_unit_hypothesis_scope_violations(
                    hypothesis,
                    facts,
                    item_path,
                )
            )
        if violations:
            raise ValueError(_format_calculation_unit_hypothesis_scope_error(violations))

    @staticmethod
    def _validate_calculation_unit_query_scope(
        queries: Sequence[str],
        calculation_unit: Mapping[str, Any],
    ) -> None:
        """拒绝 Query 对固定计算单元加入未声明的范围扩展。

        输入是已完成基础格式校验的 Query 文本和非空 calculation_unit；函数不修改
        任一输入，也不回显 Query 正文。比较阈值、容差、外部 benchmark、竞争路径、
        选择性结论，以及未在 analysis/job 字段明确声明的热化学或构象范围都不属于当前
        计算单元合同，因此抛出固定、可用于模型重答的 ``ValueError``。
        """

        if not isinstance(calculation_unit, Mapping) or not calculation_unit:
            return
        declared_text = json.dumps(
            calculation_unit,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        scope_declarations = _build_calculation_unit_scope_declarations(
            calculation_unit
        )
        violations = _collect_calculation_unit_query_scope_violations(
            queries,
            declared_text,
            scope_declarations,
        )
        if violations:
            raise ValueError(_format_calculation_unit_hypothesis_scope_error(violations))

    @staticmethod
    def _validate_query_angles(
        value: Any,
        expected_count: int,
        *,
        calculation_unit: Mapping[str, Any] | None = None,
    ) -> List[str]:
        """校验 Query 数量、词数、重复项和可选 calculation_unit 范围。

        输入是模型按 ``<>`` 分隔后的 Query 文本和本次可选的计算单元；函数只做
        确定性校验，不修改输入。返回去除首尾空白的 Query 数组；数量、词数、重复
        项或计算单元范围不合法时抛出 ``ValueError``，错误摘要不回显模型原文。
        未提供 calculation_unit 时保留历史 Query 行为。
        """

        queries = [str(item).strip() for item in value]
        if len(queries) != expected_count or any(not item for item in queries):
            raise ValueError("query angles 数量或内容非法")
        english_word_counts = [
            len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", item))
            for item in queries
        ]
        insufficient_detail = []
        for index, (item, english_word_count) in enumerate(
            zip(queries, english_word_counts), start=1
        ):
            if english_word_count >= 30:
                continue
            non_whitespace_length = len(re.sub(r"\s+", "", item))
            cjk_length = len(_QUERY_CJK_PATTERN.findall(item))
            if (
                non_whitespace_length >= _QUERY_MIN_NON_WHITESPACE_CHARS
                and cjk_length >= _QUERY_MIN_CJK_CHARS
            ):
                continue
            insufficient_detail.append(index)
        if insufficient_detail:
            raise ValueError(
                "每个 query angle 必须达到详细度要求：至少 30 个英文单词，或至少 "
                f"{_QUERY_MIN_NON_WHITESPACE_CHARS} 个非空白字符且包含至少 "
                f"{_QUERY_MIN_CJK_CHARS} 个 CJK 字符（不合格项："
                f"{insufficient_detail}）"
            )
        if len(set(queries)) != len(queries):
            raise ValueError("query angles 不得完全重复")
        if isinstance(calculation_unit, Mapping) and calculation_unit:
            HypothesisAgent._validate_calculation_unit_query_scope(
                queries,
                calculation_unit,
            )
        return queries

    def _run_indexed_hypothesis_generations_strict(
        self,
        tasks: List[Tuple[int, Any]],
        worker: Callable[[Any], Any],
    ) -> List[Tuple[int, Any]]:
        """按 Query 序号并发生成假设，并在所有已提交调用结束后失败。

        输入是带 Query 序号的独立任务和执行单个任务的函数；动作按照配置创建
        线程、等待所有已提交 Future，再按最小 Query 序号报告失败。返回值始终
        按输入顺序排列，避免完成顺序改变候选身份和后续 manifest。没有任务时
        返回空数组；任务失败时抛出原始异常，不把失败候选替换成默认文本。
        """

        if not tasks:
            return []
        try:
            worker_count = min(
                max(1, int(getattr(self, "hypothesis_generation_workers", 1))),
                len(tasks),
            )
        except (TypeError, ValueError):
            worker_count = 1
        results: Dict[int, Any] = {}
        failures: Dict[int, BaseException] = {}
        if worker_count == 1:
            for index, payload in tasks:
                try:
                    results[index] = worker(payload)
                except BaseException as exc:
                    failures[index] = exc
                    break
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="hypothesis-generation",
            ) as executor:
                futures = {
                    executor.submit(worker, payload): index
                    for index, payload in tasks
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except BaseException as exc:
                        failures[index] = exc
        if failures:
            raise failures[min(failures)]
        return [(index, results[index]) for index, _ in tasks]

    def _run_indexed_comparisons_strict(
        self,
        tasks: List[Tuple[int, Any]],
        worker: Callable[[Any], Any],
    ) -> List[Tuple[int, Any]]:
        """并行比较多组假设，同时保持串行实现原有的结果顺序。

        线程完成顺序可能不同，但主线程始终按候选原始序号收集结果。任一比较
        失败都会让整批失败，不能把失败比较自动当成“无关”或“平局”。
        """

        if not tasks:
            return []
        configured_workers = getattr(self, "comparison_workers", 1)
        if configured_workers == 0:
            worker_count = len(tasks)
        else:
            worker_count = min(configured_workers, len(tasks))
        results: Dict[int, Any] = {}
        failures: Dict[int, BaseException] = {}
        if worker_count == 1:
            for index, payload in tasks:
                try:
                    results[index] = worker(payload)
                except BaseException as exc:
                    failures[index] = exc
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="hypothesis-compare",
            ) as executor:
                futures = {
                    executor.submit(worker, payload): index
                    for index, payload in tasks
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except BaseException as exc:
                        failures[index] = exc
        if failures:
            raise failures[min(failures)]
        return [(index, results[index]) for index, _ in tasks]

    # ==================== 候选构建与去重 ====================

    @staticmethod
    def _create_hypothesis(
        hypothesis: Mapping[str, Any],
        hypothesis_id: str,
    ) -> Dict[str, Any]:
        """为两字段科学正文补充程序身份，不制造摘要或计算画像。"""

        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise ValueError("正式 hypothesis_id 必须是非空字符串")
        fields = ("strategy_name", "reasoning")
        if not isinstance(hypothesis, Mapping) or set(hypothesis) != set(fields):
            raise ValueError("正式 hypothesis 科学正文必须恰含 strategy_name 与 reasoning")
        normalized = {"hypothesis_id": hypothesis_id}
        for field in fields:
            value = hypothesis[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"正式 hypothesis.{field} 必须是非空字符串")
            normalized[field] = value.strip()
        return normalized
    @staticmethod
    def _scientific_hypothesis_projection(
        hypothesis: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """把科学正文和候选来源 Query 投影给比较模型。

        数据库 ID、既有排名和分数都不应影响模型的科学判断，因此这些字段在
        构造请求时就被移除，而不是仅靠提示词要求模型忽略。来源 Query 是候选
        科学范围的一部分；合成候选可携带多个来源，按父项顺序去重后投影。
        """

        projected = project_scientific_hypothesis_identity(hypothesis)
        source_queries = hypothesis.get(_SOURCE_QUERY_ANGLES_FIELD)
        if source_queries is None:
            source_queries = hypothesis.get("_source_query_angles")
        if source_queries is not None:
            projected[_SOURCE_QUERY_ANGLES_FIELD] = _ordered_unique_source_query_angles(
                source_queries
            )
            if not projected[_SOURCE_QUERY_ANGLES_FIELD]:
                raise ValueError("hypothesis.source_query_angles 不能为空")
        return projected

    @classmethod
    def _assert_unique_scientific_hypotheses(
        cls, hypotheses: Sequence[Mapping[str, Any]]
    ) -> None:
        """在比较前确认每个候选都有正文，且不存在逐字相同的假设。

        成对比较会消耗大量模型时间。提前拦截空项和完全重复项既避免无意义调用，
        也防止同一假设因重复出现而在胜负计分中获得额外权重。
        """

        identities = []
        bodies_present = []
        for item in hypotheses:
            projected = cls._scientific_hypothesis_projection(item)
            name = projected["strategy_name"]
            body = {
                key: value
                for key, value in projected.items()
                if key not in {"strategy_name", _SOURCE_QUERY_ANGLES_FIELD}
            }
            canonical_body = json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            identities.append((name, canonical_body))
            bodies_present.append(
                bool(str(projected.get("reasoning", "")).strip())
            )
        if any(not name for name, _ in identities) or not all(bodies_present):
            raise ValueError("hypothesis 科学正文不能为空")
        if len(set(identities)) != len(identities):
            raise ValueError("同一 batch 不得包含完全重复 hypothesis")

    @staticmethod
    def _require_source_query_provenance(
        hypotheses: Sequence[Dict[str, Any]],
    ) -> None:
        """确认比较候选都有可投影的来源 Query，并规范其顺序。

        Relationship 和 Rank 都把来源 Query 当作科学合同的一部分。缺失来源时
        不能让模型在没有 Query 对应关系的情况下静默合并或排名；非法候选直接在
        比较前失败。副作用仅限于规范化当前内部副本，不改变公开正文。
        """

        for index, hypothesis in enumerate(hypotheses):
            if not isinstance(hypothesis, dict):
                raise ValueError(f"hypotheses[{index}] 必须是内部 object")
            source_queries = _ordered_unique_source_query_angles(
                hypothesis.get(_SOURCE_QUERY_ANGLES_FIELD)
            )
            if not source_queries:
                raise ValueError(
                    f"hypotheses[{index}] 缺少 source_query_angles，不能比较"
                )
            hypothesis[_SOURCE_QUERY_ANGLES_FIELD] = source_queries

    def _classify_relationship(
        self,
        hypothesis_a: Mapping[str, Any],
        hypothesis_b: Mapping[str, Any],
        scientific_context: Mapping[str, Any],
        *,
        reserved_stage_index: Optional[int] = None,
    ) -> Dict[str, str]:
        """依据原始问题和两项科学正文判断关系，并保留审计理由。

        输入：原始 scientific_context 与 A/B 两项候选。动作：只把原始问题、两项
        两字段科学正文和各自来源 Query 交给 Relationship 阶段，比较 scope、主张、
        决定性观测、坐标签名及结果映射。副作用：不修改候选。返回：关系及非空理由；
        失败：模型协议或关系合同校验失败时由 operation runner 抛出阶段错误。
        """

        base_input = {
            "scientific_question": (
                scientific_context.get("scientific_question", "")
                if isinstance(scientific_context, Mapping)
                else ""
            ),
            "hypothesis_a": self._scientific_hypothesis_projection(hypothesis_a),
            "hypothesis_b": self._scientific_hypothesis_projection(hypothesis_b),
        }
        calculation_context = _project_calculation_unit_context(
            scientific_context.get("chemistry_context")
            if isinstance(scientific_context, Mapping)
            else None
        )
        if calculation_context:
            base_input["chemistry_context"] = calculation_context
        capability_projection = (
            scientific_context.get("chemical_capability_projection")
            if isinstance(scientific_context, Mapping)
            else None
        )
        if capability_projection:
            base_input["chemical_capability_projection"] = copy.deepcopy(
                list(capability_projection)
            )

        return self._call_model_stage(
            stage_name="relationship",
            base_input=base_input,
            specification=ModelStage(
                stage="relationship",
                stage_kind="model",
                validation_kind="json_schema",
                schema=_RELATIONSHIP_SCHEMA,
                domain_validator=_validate_relationship_payload,
                domain_validator_id="hypothesis.relationship_reasoned.v1",
            ),
            reserved_stage_index=reserved_stage_index,
        )

    def _synthesize_hypothesis_pair(
        self,
        hypothesis_a: Mapping[str, Any],
        hypothesis_b: Mapping[str, Any],
        relationship: str,
        scientific_context: Optional[Mapping[str, Any]] = None,
        relationship_reason: Optional[str] = None,
        *,
        calculation_unit: Mapping[str, Any] | None = None,
        reserved_stage_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """按关系类型和原始问题把一对假设合成为一个两字段科学正文。

        输入：A/B 科学正文、关系类型、relationship reason 和原始 scientific_context。
        动作：把原始问题、父项正文、来源 Query 及本次关系决定交给对应 synthesis
        Prompt，要求保留原题可回答性和原子能力边界。副作用：不修改父项。返回：
        strategy_name 与 reasoning；失败：关系不支持或模型协议/输出合同校验失败时抛出错误。
        """

        if relationship not in {"merge", "complement", "conflict"}:
            raise ValueError("只有非 independent 关系才能执行 synthesis")

        base_input = {
            "scientific_question": (
                scientific_context.get("scientific_question", "")
                if isinstance(scientific_context, Mapping)
                else ""
            ),
            "hypothesis_a": self._scientific_hypothesis_projection(hypothesis_a),
            "hypothesis_b": self._scientific_hypothesis_projection(hypothesis_b),
            "synthesis_output_schema": copy.deepcopy(_MERGED_HYPOTHESIS_SCHEMA),
        }
        if relationship_reason is not None:
            if not isinstance(relationship_reason, str) or not relationship_reason.strip():
                raise ValueError("relationship_reason 必须是非空字符串")
            base_input["relationship_decision"] = {
                "relationship": relationship,
                "reason": relationship_reason.strip(),
            }
        capability_projection = (
            scientific_context.get("chemical_capability_projection")
            if isinstance(scientific_context, Mapping)
            else None
        )
        if capability_projection:
            base_input["chemical_capability_projection"] = copy.deepcopy(
                list(capability_projection)
            )
        calculation_context = _project_calculation_unit_context(
            {"calculation_unit": calculation_unit}
            if isinstance(calculation_unit, Mapping) and calculation_unit
            else None
        )
        if calculation_context:
            base_input["chemistry_context"] = calculation_context
        domain_validator = (
            _CalculationUnitSynthesisValidator(
                calculation_context["calculation_unit"]
            )
            if calculation_context
            else _validate_synthesis_payload
        )

        return self._call_model_stage(
            stage_name=relationship,
            base_input=base_input,
            specification=ModelStage(
                stage=relationship,
                stage_kind="model",
                validation_kind="json_schema",
                schema=_MERGED_HYPOTHESIS_SCHEMA,
                domain_validator=domain_validator,
                domain_validator_id=f"hypothesis.{relationship}_two_field.v1",
            ),
            reserved_stage_index=reserved_stage_index,
        )

    def _merge_hypothesis_pair(
        self,
        hypothesis_a: Mapping[str, Any],
        hypothesis_b: Mapping[str, Any],
        scientific_context: Mapping[str, Any],
        machine_tool_identifiers: Sequence[str],
        *,
        calculation_unit: Mapping[str, Any] | None = None,
        reserved_stage_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """兼容旧调用点，将 merge 委托给两字段 synthesis。"""

        del machine_tool_identifiers
        resolved_calculation_unit = calculation_unit
        if resolved_calculation_unit is None:
            resolved_calculation_unit = _project_calculation_unit_context(
                scientific_context.get("chemistry_context")
                if isinstance(scientific_context, Mapping)
                else None
            ).get("calculation_unit")
        return self._synthesize_hypothesis_pair(
            hypothesis_a,
            hypothesis_b,
            "merge",
            scientific_context,
            calculation_unit=resolved_calculation_unit,
            reserved_stage_index=reserved_stage_index,
        )

    def _optimize_hypotheses(
        self,
        hypotheses: Sequence[Mapping[str, Any]],
        scientific_context: Mapping[str, Any],
        merge_hypothesis_ids: Sequence[str],
        machine_tool_identifiers: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """对跨 Query 的候选执行一轮全量 pair 关系判断与 merge 归约。

        所有四类关系判断先按快照顺序完成。``merge`` 边按确定性的 complete-link
        规则形成互不重叠的组：新成员必须与组内每个父项都被判为 merge，避免一条
        非传递边把不同机理串在一起。每组顺序合成为唯一候选，每个原候选至多消费
        一次；``complement``、``conflict`` 和 ``independent`` 只保留模型审计。
        """

        candidates = [copy.deepcopy(dict(item)) for item in hypotheses]
        self._require_source_query_provenance(candidates)
        self._assert_unique_scientific_hypotheses(candidates)
        merge_ids = list(merge_hypothesis_ids)
        del machine_tool_identifiers
        calculation_unit = _project_calculation_unit_context(
            scientific_context.get("chemistry_context")
            if isinstance(scientific_context, Mapping)
            else None
        ).get("calculation_unit")
        if len(candidates) < 2:
            return candidates

        pairs = list(combinations(range(len(candidates)), 2))
        if not pairs:
            return candidates
        relationship_slots: list[int] = []
        if self._operation_runner is not None:
            relationship_slots = self._operation_runner.reserve_model_stages(
                [
                    ModelStage(
                        stage="relationship",
                        stage_kind="model",
                        validation_kind="json_schema",
                        schema=_RELATIONSHIP_SCHEMA,
                        domain_validator=_validate_relationship_payload,
                        domain_validator_id="hypothesis.relationship_reasoned.v1",
                    )
                    for _ in pairs
                ]
            )
        relationship_tasks = [
            (
                pair_index,
                (
                    candidates[left],
                    candidates[right],
                    relationship_slots[pair_index]
                    if relationship_slots
                    else None,
                ),
            )
            for pair_index, (left, right) in enumerate(pairs)
        ]

        def classify_pair(payload: Any) -> Dict[str, str]:
            left, right, reserved_stage_index = payload
            return self._classify_relationship(
                left,
                right,
                scientific_context,
                reserved_stage_index=reserved_stage_index,
            )

        relationship_results = self._run_indexed_comparisons_strict(
            relationship_tasks,
            classify_pair,
        )
        merge_relationships = [
            (pair_index, pairs[pair_index], result)
            for pair_index, result in relationship_results
            if result["relationship"] == "merge"
        ]

        merge_lookup = {
            tuple(sorted((left, right))): (pair_index, result)
            for pair_index, (left, right), result in merge_relationships
        }
        candidate_groups: list[list[int]] = []
        for candidate_index in range(len(candidates)):
            for group in candidate_groups:
                if all(
                    tuple(sorted((candidate_index, member))) in merge_lookup
                    for member in group
                ):
                    group.append(candidate_index)
                    break
            else:
                candidate_groups.append([candidate_index])
        merge_groups = [group for group in candidate_groups if len(group) > 1]

        synthesis_count = sum(len(group) - 1 for group in merge_groups)
        synthesis_slots: list[int] = []
        synthesis_domain_validator = (
            _CalculationUnitSynthesisValidator(calculation_unit)
            if calculation_unit
            else _validate_synthesis_payload
        )
        if synthesis_count and self._operation_runner is not None:
            synthesis_slots = self._operation_runner.reserve_model_stages(
                [
                    ModelStage(
                        stage="merge",
                        stage_kind="model",
                        validation_kind="json_schema",
                        schema=_MERGED_HYPOTHESIS_SCHEMA,
                        domain_validator=synthesis_domain_validator,
                        domain_validator_id="hypothesis.merge_two_field.v1",
                    )
                    for _ in range(synthesis_count)
                ]
            )

        synthesis_cursor = 0
        group_tasks = []
        for group_index, group in enumerate(merge_groups):
            fold_count = len(group) - 1
            if synthesis_cursor + fold_count > len(merge_ids):
                raise ValueError("Controller 预分配的 synthesis hypothesis_id 不足")
            group_ids = merge_ids[synthesis_cursor : synthesis_cursor + fold_count]
            group_slots = (
                synthesis_slots[synthesis_cursor : synthesis_cursor + fold_count]
                if synthesis_slots
                else [None] * fold_count
            )
            source_records = []
            for left, right in combinations(group, 2):
                pair_index, relation_result = merge_lookup[(left, right)]
                source_records.append(
                    {
                        "pair_index": pair_index,
                        "relationship": "merge",
                        "reason": relation_result["reason"],
                        "parent_hypothesis_ids": [
                            candidates[left]["hypothesis_id"],
                            candidates[right]["hypothesis_id"],
                        ],
                    }
                )
            fold_reasons = []
            previous_members = [group[0]]
            for candidate_index in group[1:]:
                fold_reasons.append(
                    "；".join(
                        merge_lookup[
                            tuple(sorted((candidate_index, member)))
                        ][1]["reason"]
                        for member in previous_members
                    )
                )
                previous_members.append(candidate_index)
            group_tasks.append(
                (
                    group_index,
                    (
                        group,
                        group_ids,
                        group_slots,
                        fold_reasons,
                        source_records,
                    ),
                )
            )
            synthesis_cursor += fold_count

        def synthesize_group(payload: Any) -> Dict[str, Any]:
            group, group_ids, group_slots, fold_reasons, source_records = payload
            current = copy.deepcopy(candidates[group[0]])
            source_query_angles = _ordered_unique_source_query_angles(
                current.get(_SOURCE_QUERY_ANGLES_FIELD)
            )
            for fold_index, candidate_index in enumerate(group[1:]):
                raw_synthesis = self._synthesize_hypothesis_pair(
                    current,
                    candidates[candidate_index],
                    "merge",
                    scientific_context,
                    relationship_reason=fold_reasons[fold_index],
                    calculation_unit=calculation_unit,
                    reserved_stage_index=group_slots[fold_index],
                )
                current = self._create_hypothesis(
                    raw_synthesis,
                    group_ids[fold_index],
                )
                source_query_angles = _ordered_unique_source_query_angles(
                    source_query_angles,
                    candidates[candidate_index].get(_SOURCE_QUERY_ANGLES_FIELD),
                )
                if source_query_angles:
                    current[_SOURCE_QUERY_ANGLES_FIELD] = source_query_angles
            current["synthesis_sources"] = copy.deepcopy(source_records)
            return current

        synthesis_results = self._run_indexed_comparisons_strict(
            group_tasks,
            synthesize_group,
        )
        if len(synthesis_results) != len(merge_groups):
            raise ValueError("所有 merge 组都必须完成 synthesis")

        consumed_indices = {
            index
            for group in merge_groups
            for index in group
        }
        optimized = [
            copy.deepcopy(candidate)
            for index, candidate in enumerate(candidates)
            if index not in consumed_indices
        ]
        synthesis_by_identity: dict[str, Dict[str, Any]] = {}
        synthesis_order: list[str] = []
        for _, normalized in synthesis_results:
            identity = json.dumps(
                {
                    key: value
                    for key, value in self._scientific_hypothesis_projection(
                        normalized
                    ).items()
                    if key != _SOURCE_QUERY_ANGLES_FIELD
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            existing = synthesis_by_identity.get(identity)
            if existing is None:
                synthesis_by_identity[identity] = normalized
                synthesis_order.append(identity)
            else:
                merged_sources = _ordered_unique_source_query_angles(
                    existing.get(_SOURCE_QUERY_ANGLES_FIELD),
                    normalized.get(_SOURCE_QUERY_ANGLES_FIELD),
                )
                if merged_sources:
                    existing[_SOURCE_QUERY_ANGLES_FIELD] = merged_sources
                existing.setdefault("synthesis_sources", []).extend(
                    copy.deepcopy(normalized.get("synthesis_sources", []))
                )
        optimized.extend(synthesis_by_identity[key] for key in synthesis_order)
        self._assert_unique_scientific_hypotheses(optimized)
        return optimized

    # ==================== 候选优化与排名 ====================

    def _rank_strategies(
        self,
        strategies: Sequence[Mapping[str, Any]],
        scientific_context: Mapping[str, Any],
        stage_tool_projection: Sequence[Mapping[str, Any]] = (),
        chemical_capability_projection: Sequence[Mapping[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        """让最终候选集合中的每一对直接比较，再按胜负差排序。

        比较可以并行完成；最终仍按候选原始位置打破同分，因此同一组模型回答
        无论线程先后如何都得到相同排名。
        """

        candidates = [copy.deepcopy(dict(item)) for item in strategies]
        self._require_source_query_provenance(candidates)
        self._assert_unique_scientific_hypotheses(candidates)
        if not candidates:
            raise ValueError("ranked_strategies 不能为空")
        if len(candidates) == 1:
            only = candidates[0]
            only.update({"wins": 0, "losses": 0, "score": 0, "rank": 1})
            return [only]
        pairs = list(combinations(range(len(candidates)), 2))
        # 每一对候选在提交并发比较前先记住原始位置。结果即使乱序返回，也会按这
        # 个位置累加胜负，网络快慢不会改变最终排名。
        comparison_tasks = list(enumerate(pairs))
        reserved_rank_slots: list[int] = []
        if self._operation_runner is not None:
            reserved_rank_slots = self._operation_runner.reserve_model_stages(
                [
                    ModelStage(
                        stage="rank",
                        stage_kind="model",
                        validation_kind="json_schema",
                        schema=_RANK_SCHEMA,
                        domain_validator=_validate_rank_payload,
                        domain_validator_id="hypothesis.rank_reasoned.v1",
                    )
                    for _ in comparison_tasks
                ]
            )
            comparison_tasks = [
                (pair_index, (pair, reserved_rank_slots[pair_index]))
                for pair_index, pair in comparison_tasks
            ]
        def compare_pair(pair_input: Any) -> Dict[str, str]:
            if reserved_rank_slots:
                pair, reserved_stage_index = pair_input
            else:
                pair = pair_input
                reserved_stage_index = None
            left, right = pair
            base_input = {
                "scientific_question": scientific_context.get(
                    "scientific_question", ""
                ),
                "hypothesis_a": self._scientific_hypothesis_projection(
                    candidates[left]
                ),
                "hypothesis_b": self._scientific_hypothesis_projection(
                    candidates[right]
                ),
                "stage_tool_projection": copy.deepcopy(
                    list(stage_tool_projection)
                ),
            }
            if chemical_capability_projection:
                base_input["chemical_capability_projection"] = copy.deepcopy(
                    list(chemical_capability_projection)
                )
            calculation_context = _project_calculation_unit_context(
                scientific_context.get("chemistry_context")
                if isinstance(scientific_context, Mapping)
                else None
            )
            if calculation_context:
                base_input["chemistry_context"] = calculation_context
            return self._call_model_stage(
                stage_name="rank",
                base_input=base_input,
                specification=ModelStage(
                    stage="rank",
                    stage_kind="model",
                    validation_kind="json_schema",
                    schema=_RANK_SCHEMA,
                    domain_validator=_validate_rank_payload,
                    domain_validator_id="hypothesis.rank_reasoned.v1",
                ),
                reserved_stage_index=reserved_stage_index,
            )

        comparison_results = self._run_indexed_comparisons_strict(
            comparison_tasks,
            compare_pair,
        )
        wins = [0] * len(candidates)
        losses = [0] * len(candidates)
        for pair_index, ranking in comparison_results:
            left, right = pairs[pair_index]
            winner = ranking["winner"]
            winning_index, losing_index = (
                (left, right) if winner == "A" else (right, left)
            )
            wins[winning_index] += 1
            losses[losing_index] += 1
        ranked = []
        for index, candidate in enumerate(candidates):
            candidate.update(
                {
                    "wins": wins[index],
                    "losses": losses[index],
                    "score": wins[index] - losses[index],
                    "_input_index": index,
                }
            )
            ranked.append(candidate)
        ranked.sort(
            key=lambda item: (
                -item["score"],
                -item["wins"],
                item["_input_index"],
            )
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate.pop("_input_index")
            candidate["rank"] = rank
        return ranked

    # ==================== 批次与再生成材料校验 ====================

    @staticmethod
    def _validate_batch_inputs(
        *,
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        hypothesis_ids_by_query: Sequence[Sequence[str]],
        merge_hypothesis_ids: Sequence[str],
    ) -> None:
        """在调用模型前确认候选数量与控制器预先分配的身份一一对应。

        这里同时检查每个研究角度的名额、合并候选的备用身份和最终入选数，尽早
        发现缺号、重号或配置不一致，避免模型生成完才发现结果无法安全写回。
        """

        for name, value in (
            ("num_queries", num_queries),
            ("num_hypotheses_per_query", num_hypotheses_per_query),
            ("top_n", top_n),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if top_n > num_queries * num_hypotheses_per_query:
            raise ValueError("top_n 不能超过初始候选总数")
        if len(hypothesis_ids_by_query) != num_queries or any(
            len(group) != num_hypotheses_per_query
            for group in hypothesis_ids_by_query
        ):
            raise ValueError("hypothesis_ids_by_query shape 与冻结数量不一致")
        structure_ids = [item for group in hypothesis_ids_by_query for item in group]
        all_ids = structure_ids + list(merge_hypothesis_ids)
        if any(not isinstance(item, str) or not item.strip() for item in all_ids):
            raise ValueError("Controller分配的 hypothesis_id 必须是非空字符串")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("同一 batch 的 hypothesis_id 必须唯一")

    @staticmethod
    def _require_exact_regeneration_object(
        value: Any,
        expected_keys: set[str],
        label: str,
    ) -> Dict[str, Any]:
        """确认一层历史对象既不缺信息，也没有夹带本阶段未允许的内容。

        再生成历史会直接进入模型上下文。若只挑选已知字段而忽略多余内容，调用方
        可能无意中发送路径、内部状态或未入选正文，破坏 Memory A/B 的信息边界。
        """

        if not isinstance(value, Mapping):
            raise ValueError(f"{label} 必须是 object")
        actual_keys = set(value)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"{label} 字段不完整；缺少={missing}，多余={extra}"
            )
        return copy.deepcopy(dict(value))

    @classmethod
    def _validate_regeneration_package(
        cls,
        value: Any,
        *,
        memory_enabled: bool,
        expected_task_id: str | None = None,
        expected_batch_id: str | None = None,
    ) -> Dict[str, Any]:
        """验证下一批模型可见的历史范围和固定生成配置。

        旧 workflow 清单仍可使用 top_n_workflows 别名；活动 Investigation v2
        的 top_n_hypotheses 只在 operation 入口转换，不能在这里混入旧公共 package。
        """

        del cls
        return _validate_regeneration_package_payload(
            value,
            memory_enabled=memory_enabled,
            expected_task_id=expected_task_id,
            expected_batch_id=expected_batch_id,
        )

    # ==================== 假设批次运行 ====================

    @staticmethod
    def _prepare_hypothesis_batch_context(
        *,
        scientific_question: str,
        stage_tool_projection: Sequence[Mapping[str, Any]],
        initial_retrieval_phase: Mapping[str, Any],
        chemistry_context: Mapping[str, Any],
        supplementary_scientific_input: Optional[Mapping[str, Any]],
    ) -> tuple[
        list[Dict[str, Any]],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        """校验批次上下文并整理模型可见的工具、检索和计算单元事实。"""

        if not isinstance(scientific_question, str) or not scientific_question.strip():
            raise ValueError("scientific_question 必须是非空字符串")
        normalized_projection = (
            HypothesisAgent._validate_stage_tool_projection(
                list(stage_tool_projection)
            )
            if stage_tool_projection
            else []
        )
        if not isinstance(initial_retrieval_phase, Mapping) or not isinstance(
            chemistry_context, Mapping
        ):
            raise ValueError("Retrieval 与 chemistry context 必须是 object")
        provided_context = _project_calculation_unit_context(
            supplementary_scientific_input
            if supplementary_scientific_input is not None
            else chemistry_context
        )
        retrieval_context = initial_retrieval_phase.get("chemistry_context")
        if not isinstance(retrieval_context, Mapping):
            retrieval_context = chemistry_context
        retrieval_projection = {
            "keywords": copy.deepcopy(initial_retrieval_phase.get("keywords", [])),
            "literature_review": copy.deepcopy(
                initial_retrieval_phase.get("literature_review", "")
            ),
            "mechanistic_clues": copy.deepcopy(
                initial_retrieval_phase.get("mechanistic_clues", []) or []
            ),
            "limitations": copy.deepcopy(
                initial_retrieval_phase.get("limitations", []) or []
            ),
            "chemistry_context": copy.deepcopy(retrieval_context or {}),
        }
        return normalized_projection, provided_context, retrieval_projection

    def _resolve_hypothesis_batch_queries(
        self,
        *,
        scientific_question: str,
        initial_query_angles: Optional[Sequence[str]],
        regeneration_package: Optional[Mapping[str, Any]],
        num_queries: int,
        provided_calculation_context: Mapping[str, Any],
        retrieval_projection: Mapping[str, Any],
    ) -> list[str]:
        """生成首批研究角度或复用后续批冻结角度。"""

        if initial_query_angles is not None:
            return self._validate_query_angles(
                list(initial_query_angles),
                num_queries,
            )
        if regeneration_package is not None:
            raise ValueError("regeneration 必须复用冻结 initial_query_angles")
        query_calculation_unit = provided_calculation_context.get(
            "calculation_unit"
        )
        raw_queries = self._call_model_stage(
            stage_name="query",
            base_input={
                "scientific_question": scientific_question.strip(),
                "num_queries": num_queries,
                "literature_review": copy.deepcopy(
                    retrieval_projection.get("literature_review", "")
                ),
                "mechanistic_clues": copy.deepcopy(
                    retrieval_projection.get("mechanistic_clues", [])
                ),
                "limitations": copy.deepcopy(
                    retrieval_projection.get("limitations", [])
                ),
                "chemistry_context": copy.deepcopy(
                    retrieval_projection.get("chemistry_context", {})
                ),
                **(
                    {"chemistry_context": provided_calculation_context}
                    if provided_calculation_context
                    else {}
                ),
            },
            specification=ModelStage(
                stage="query",
                stage_kind="model",
                validation_kind="query_delimited_text",
                expected_count=num_queries,
                domain_validator=lambda value: self._validate_query_angles(
                    value,
                    num_queries,
                    calculation_unit=query_calculation_unit,
                ),
                domain_validator_id="hypothesis.query_angles.v1",
            ),
        )
        return list(raw_queries)

    def _validate_hypothesis_batch_regeneration(
        self,
        *,
        regeneration_package: Optional[Mapping[str, Any]],
        memory_enabled: Optional[bool],
        expected_task_id: str | None,
        expected_batch_id: str | None,
        scientific_question: str,
        queries: Sequence[str],
        initial_retrieval_phase: Mapping[str, Any],
        chemistry_context: Mapping[str, Any],
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        provided_calculation_context: Mapping[str, Any],
    ) -> tuple[str, Optional[Dict[str, Any]], Mapping[str, Any], Dict[str, Any]]:
        """校验后续批冻结上下文并返回模型阶段、历史包和有效化学条件。"""

        if regeneration_package is None:
            if memory_enabled is not None:
                raise ValueError("初始 batch 不接受 memory_enabled")
            return "hypothesis", None, chemistry_context, dict(provided_calculation_context)
        if not isinstance(memory_enabled, bool):
            raise ValueError("regeneration 必须显式提供 memory_enabled")
        validated_package = self._validate_regeneration_package(
            regeneration_package,
            memory_enabled=memory_enabled,
            expected_task_id=expected_task_id,
            expected_batch_id=expected_batch_id,
        )
        frozen_context = validated_package["scientific_context"]
        if frozen_context["scientific_question"] != scientific_question.strip():
            raise ValueError("regeneration package 的 scientific question 已漂移")
        if frozen_context["initial_query_angles"] != list(queries):
            raise ValueError("regeneration package 的 query angles 已漂移")
        expected_context = project_hypothesis_regeneration_context(
            scientific_question=scientific_question.strip(),
            initial_query_angles=list(queries),
            initial_retrieval_phase=initial_retrieval_phase,
            chemistry_context=chemistry_context,
        )
        if frozen_context != expected_context:
            raise ValueError("regeneration package 的冻结科学上下文已漂移")
        if validated_package["generation_config"] != {
            "num_queries": num_queries,
            "num_hypotheses_per_query": num_hypotheses_per_query,
            "top_n": top_n,
        }:
            raise ValueError("regeneration package 的 generation_config 已漂移")
        frozen_chemistry_context = copy.deepcopy(
            dict(frozen_context["chemistry_context"])
        )
        frozen_calculation_context = _project_calculation_unit_context(
            frozen_chemistry_context
        )
        return (
            "regeneration",
            validated_package,
            frozen_chemistry_context,
            frozen_calculation_context,
        )

    def _build_hypothesis_generation_input(
        self,
        *,
        generation_stage: str,
        validated_regeneration_package: Optional[Mapping[str, Any]],
        scientific_question: str,
        query: str,
        num_hypotheses_per_query: int,
        normalized_tool_projection: Sequence[Mapping[str, Any]],
        chemical_capability_projection: Sequence[Mapping[str, Any]],
        retrieval_projection: Mapping[str, Any],
        hypothesis_schema: Mapping[str, Any],
        provided_calculation_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """构造单个 query 的 Hypothesis 或 regeneration 模型输入。"""

        if generation_stage == "hypothesis":
            generation_input: Dict[str, Any] = {
                "scientific_question": scientific_question.strip(),
                "query_angle": query,
                "literature_review": copy.deepcopy(
                    retrieval_projection.get("literature_review", "")
                ),
                "mechanistic_clues": copy.deepcopy(
                    retrieval_projection.get("mechanistic_clues", [])
                ),
                "limitations": copy.deepcopy(
                    retrieval_projection.get("limitations", [])
                ),
                "chemistry_context": copy.deepcopy(
                    retrieval_projection.get("chemistry_context", {})
                ),
                "num_hypotheses": num_hypotheses_per_query,
                "stage_tool_projection": copy.deepcopy(
                    list(normalized_tool_projection)
                ),
                "hypothesis_output_schema": copy.deepcopy(hypothesis_schema),
            }
        else:
            generation_input = copy.deepcopy(
                dict(validated_regeneration_package or {})
            )
            generation_input.pop("source_record_ids", None)
            generation_input.update(
                {
                    "scientific_question": scientific_question.strip(),
                    "current_query": query,
                    "query_angle": query,
                    "num_hypotheses": num_hypotheses_per_query,
                    "stage_tool_projection": copy.deepcopy(
                        list(normalized_tool_projection)
                    ),
                    "hypothesis_output_schema": copy.deepcopy(hypothesis_schema),
                }
            )
        if chemical_capability_projection:
            generation_input["chemical_capability_projection"] = copy.deepcopy(
                list(chemical_capability_projection)
            )
        if provided_calculation_context:
            generation_input["chemistry_context"] = copy.deepcopy(
                dict(provided_calculation_context)
            )
        return generation_input

    def _generate_hypothesis_batch_groups(
        self,
        *,
        generation_stage: str,
        queries: Sequence[str],
        scientific_question: str,
        normalized_tool_projection: Sequence[Mapping[str, Any]],
        chemical_capability_projection: Sequence[Mapping[str, Any]],
        retrieval_projection: Mapping[str, Any],
        provided_calculation_context: Mapping[str, Any],
        validated_regeneration_package: Optional[Mapping[str, Any]],
        num_hypotheses_per_query: int,
        hypothesis_ids_by_query: Sequence[Sequence[str]],
        hypothesis_schema: Mapping[str, Any],
        machine_tool_identifiers: Sequence[str],
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        """逐研究角度调用模型并建立带正式身份的候选分组。"""

        calculation_unit = provided_calculation_context.get("calculation_unit")
        hypothesis_domain_validator = (
            _CalculationUnitHypothesisBatchValidator(calculation_unit)
            if isinstance(calculation_unit, Mapping) and calculation_unit
            else lambda value: self._normalize_generated_hypotheses(
                value,
                machine_tool_identifiers,
            )
        )
        generation_specs: list[ModelStage] = []
        generation_tasks: list[tuple[int, Any]] = []
        # 每个 Query 先独立构造输入和校验规格，再按稳定顺序预留 manifest
        # stage。线程只负责模型调用和响应校验，正式 hypothesis_id 仍由主线程
        # 按 Query 顺序补入，避免完成顺序影响输出。
        for query_index, query in enumerate(queries, start=1):
            generation_input = self._build_hypothesis_generation_input(
                generation_stage=generation_stage,
                validated_regeneration_package=validated_regeneration_package,
                scientific_question=scientific_question,
                query=query,
                num_hypotheses_per_query=num_hypotheses_per_query,
                normalized_tool_projection=normalized_tool_projection,
                chemical_capability_projection=chemical_capability_projection,
                retrieval_projection=retrieval_projection,
                hypothesis_schema=hypothesis_schema,
                provided_calculation_context=provided_calculation_context,
            )
            if generation_stage == "hypothesis":
                generation_input["literature_review"] = retrieval_projection[
                    "literature_review"
                ]
            generation_spec = ModelStage(
                stage=generation_stage,
                stage_kind="model",
                validation_kind="json_schema",
                query_index=query_index,
                schema=hypothesis_schema,
                domain_validator=hypothesis_domain_validator,
                domain_validator_id="hypothesis.generated_single_claim.v4",
            )
            generation_specs.append(generation_spec)
            # 每个 worker 都拿到独立的输入深拷贝，provider 或消息构造器修改本地
            # 可变对象时不会污染同批的其他 Query。
            generation_tasks.append(
                (
                    query_index,
                    (
                        copy.deepcopy(generation_input),
                        generation_spec,
                        None,
                    ),
                )
            )

        try:
            generation_worker_count = min(
                max(1, int(getattr(self, "hypothesis_generation_workers", 1))),
                len(generation_specs),
            )
        except (TypeError, ValueError):
            generation_worker_count = 1
        generation_slots: list[int] = []
        # 并发模式必须先一次性预留全部 stage；串行默认继续按需预留，保留原有
        # 首个失败即停止且不留下未执行 stage 的行为。
        if self._operation_runner is not None and generation_worker_count > 1:
            generation_slots = self._operation_runner.reserve_model_stages(
                generation_specs
            )
            generation_tasks = [
                (
                    query_index,
                    (
                        copy.deepcopy(payload[0]),
                        payload[1],
                        generation_slots[query_index - 1],
                    ),
                )
                for query_index, payload in generation_tasks
            ]

        def generate_for_query(payload: Any) -> Any:
            generation_input, generation_spec, reserved_stage_index = payload
            return self._call_model_stage(
                stage_name=generation_stage,
                base_input=copy.deepcopy(generation_input),
                specification=generation_spec,
                reserved_stage_index=reserved_stage_index,
            )

        generated_results = self._run_indexed_hypothesis_generations_strict(
            generation_tasks,
            generate_for_query,
        )
        hypotheses_by_query: list[Dict[str, Any]] = []
        all_hypotheses: list[Dict[str, Any]] = []
        for query_index, raw_hypotheses in generated_results:
            query = queries[query_index - 1]
            structured = [
                self._create_hypothesis(hypothesis, hypothesis_id)
                for hypothesis, hypothesis_id in zip(
                    copy.deepcopy(list(raw_hypotheses)),
                    hypothesis_ids_by_query[query_index - 1],
                )
            ]
            for candidate in structured:
                # 来源 Query 只在内部关系、合成和排名链路中保留；最终公开投影会
                # 移除该字段，维持既有两字段 hypothesis 输出 schema。
                candidate[_SOURCE_QUERY_ANGLES_FIELD] = [query]
            hypotheses_by_query.append({"query": query, "hypotheses": structured})
            all_hypotheses.extend(structured)
        return hypotheses_by_query, all_hypotheses

    def generate_hypothesis_batch(
        self,
        *,
        scientific_question: str,
        stage_tool_projection: Sequence[Mapping[str, Any]],
        chemical_capability_projection: Sequence[Mapping[str, Any]] | None = None,
        initial_query_angles: Optional[Sequence[str]],
        initial_retrieval_phase: Mapping[str, Any],
        chemistry_context: Mapping[str, Any],
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        hypothesis_ids_by_query: Sequence[Sequence[str]],
        merge_hypothesis_ids: Sequence[str],
        regeneration_package: Optional[Mapping[str, Any]] = None,
        memory_enabled: Optional[bool] = None,
        case_profile: str | None = None,
        supplementary_scientific_input: Optional[Mapping[str, Any]] = None,
        expected_task_id: str | None = None,
        expected_batch_id: str | None = None,
        task_config_schema: Literal["legacy_workflow_v1", "investigation_v2"] = (
            "legacy_workflow_v1"
        ),
    ) -> Dict[str, Any]:
        """生成一整批新假设，完成结构化、同义合并和第一次排名。

        首批会先生成研究角度；后续批直接复用首批角度和检索结果，只重新生成
        假设正文。所有正式身份均由调用方提供，模型从未看到这些身份。若输入含
        Investigation Working Memory，调用方还必须显式提供 expected task/batch 身份，
        否则在模型调用前失败。
        """

        self._validate_batch_inputs(
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            top_n=top_n,
            hypothesis_ids_by_query=hypothesis_ids_by_query,
            merge_hypothesis_ids=merge_hypothesis_ids,
        )
        self._validate_hypothesis_operation_schema(task_config_schema, top_n)
        normalized_projection, provided_context, retrieval_projection = (
            self._prepare_hypothesis_batch_context(
                scientific_question=scientific_question,
                stage_tool_projection=stage_tool_projection,
                initial_retrieval_phase=initial_retrieval_phase,
                chemistry_context=chemistry_context,
                supplementary_scientific_input=supplementary_scientific_input,
            )
        )
        normalized_capability_projection = (
            copy.deepcopy(list(chemical_capability_projection))
            if chemical_capability_projection
            else []
        )
        queries = self._resolve_hypothesis_batch_queries(
            scientific_question=scientific_question,
            initial_query_angles=initial_query_angles,
            regeneration_package=regeneration_package,
            num_queries=num_queries,
            provided_calculation_context=provided_context,
            retrieval_projection=retrieval_projection,
        )
        (
            generation_stage,
            validated_package,
            effective_chemistry_context,
            effective_calculation_context,
        ) = self._validate_hypothesis_batch_regeneration(
            regeneration_package=regeneration_package,
            memory_enabled=memory_enabled,
            expected_task_id=expected_task_id,
            expected_batch_id=expected_batch_id,
            scientific_question=scientific_question,
            queries=queries,
            initial_retrieval_phase=initial_retrieval_phase,
            chemistry_context=chemistry_context,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            top_n=top_n,
            provided_calculation_context=provided_context,
        )
        if validated_package is not None:
            # package 内部可以暂存 top_n，但 provider 必须收到调用方明确选择的
            # legacy 或 Investigation v2 字段，不能在 prompt 层猜 schema 来源。
            validated_package["generation_config"] = (
                self._build_hypothesis_operation_generation_config(
                    num_queries=num_queries,
                    num_hypotheses_per_query=num_hypotheses_per_query,
                    top_n=top_n,
                    task_config_schema=task_config_schema,
                )
            )
        hypothesis_schema = _hypothesis_array_schema(num_hypotheses_per_query)
        hypotheses_by_query, all_hypotheses = self._generate_hypothesis_batch_groups(
            generation_stage=generation_stage,
            queries=queries,
            scientific_question=scientific_question,
            normalized_tool_projection=normalized_projection,
            chemical_capability_projection=normalized_capability_projection,
            retrieval_projection=retrieval_projection,
            provided_calculation_context=effective_calculation_context,
            validated_regeneration_package=validated_package,
            num_hypotheses_per_query=num_hypotheses_per_query,
            hypothesis_ids_by_query=hypothesis_ids_by_query,
            hypothesis_schema=hypothesis_schema,
            machine_tool_identifiers=[],
        )
        comparison_context = copy.deepcopy(dict(effective_chemistry_context))
        if effective_calculation_context:
            comparison_context.update(copy.deepcopy(dict(effective_calculation_context)))
        scientific_context = {
            "scientific_question": scientific_question.strip(),
            "initial_query_angles": list(queries),
            "chemistry_context": comparison_context,
            "chemical_capability_projection": copy.deepcopy(
                normalized_capability_projection
            ),
        }
        optimized = self._optimize_hypotheses(
            all_hypotheses,
            scientific_context,
            merge_hypothesis_ids,
            [],
        )
        ranked = self._rank_strategies(
            optimized,
            scientific_context,
            normalized_projection,
            chemical_capability_projection=normalized_capability_projection,
        )
        return {
            "queries": list(queries),
            "hypotheses_by_query": [
                {
                    "query": group["query"],
                    "hypotheses": [
                        RankedHypothesisGenerator._project_hypothesis(item)
                        for item in group["hypotheses"]
                    ],
                }
                for group in hypotheses_by_query
            ],
            "ranked_strategies": [
                RankedHypothesisGenerator._project_hypothesis(item)
                for item in ranked
            ],
        }

    @staticmethod
    def _validate_hypothesis_operation_schema(
        task_config_schema: str,
        top_n: int,
    ) -> None:
        """校验 Hypothesis operation 使用的清单 schema 和 Top N。"""

        if task_config_schema not in {"legacy_workflow_v1", "investigation_v2"}:
            raise ValueError(
                "task_config_schema 必须是 legacy_workflow_v1 或 investigation_v2"
            )
        if task_config_schema == "investigation_v2" and top_n != 5:
            raise ValueError("Investigation v2 的 top_n_hypotheses 必须严格为 5")

    @staticmethod
    def _build_hypothesis_operation_generation_config(
        *,
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        task_config_schema: str,
    ) -> Dict[str, Any]:
        """按 schema 构造 operation 清单中的生成数量字段。"""

        config = {
            "num_queries": num_queries,
            "num_hypotheses_per_query": num_hypotheses_per_query,
        }
        field_name = (
            "top_n_hypotheses"
            if task_config_schema == "investigation_v2"
            else "top_n_workflows"
        )
        config[field_name] = top_n
        return config

    @staticmethod
    def _normalize_hypothesis_replay_base_input(
        base_input: Mapping[str, Any],
        *,
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        task_config_schema: Literal["legacy_workflow_v1", "investigation_v2"],
    ) -> Dict[str, Any]:
        """把旧 replay seam 的内部 ``top_n`` 转成明确的清单字段。

        生产 operation 只写入 ``top_n_workflows`` 或
        ``top_n_hypotheses``。少数迁移期调用方仍会从已校验 package 传入旧的
        内部键；这里仅在 finalized manifest 查找前复制并归一化该兼容输入，
        不为新的请求猜测 schema，也不改变 manifest 中已经明确的字段。
        """

        normalized = copy.deepcopy(dict(base_input))
        generation_config = normalized.get("generation_config")
        if not isinstance(generation_config, Mapping):
            return normalized
        if set(generation_config) != {
            "num_queries",
            "num_hypotheses_per_query",
            "top_n",
        }:
            return normalized
        normalized["generation_config"] = (
            HypothesisAgent._build_hypothesis_operation_generation_config(
                num_queries=num_queries,
                num_hypotheses_per_query=num_hypotheses_per_query,
                top_n=top_n,
                task_config_schema=task_config_schema,
            )
        )
        return normalized

    @staticmethod
    def _prepare_initial_hypothesis_operation_base_input(
        *,
        initial_query_angles: Optional[Sequence[str]],
        initial_retrieval_phase: Optional[Mapping[str, Any]],
        chemistry_context: Mapping[str, Any],
        scientific_question: str,
        generation_config: Mapping[str, Any],
        normalized_tool_projection: Sequence[Mapping[str, Any]],
        chemical_capability_projection: Sequence[Mapping[str, Any]] = (),
        memory_enabled: Optional[bool],
        retrieval_request: Optional[Mapping[str, Any]],
        retrieve: Optional[Callable[[Callable[[Mapping[str, Any]], None]], Any]],
        supplementary_scientific_input: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """校验首批入口并构造冻结 Retrieval 或恢复请求的 base input。"""

        if initial_query_angles is not None:
            raise ValueError("首批 query 必须由假设操作现场生成")
        has_frozen_retrieval = isinstance(initial_retrieval_phase, Mapping)
        has_retrieval_runner = isinstance(retrieval_request, Mapping) and callable(
            retrieve
        )
        if has_frozen_retrieval == has_retrieval_runner:
            raise ValueError(
                "首批必须且只能提供冻结 Retrieval 或历史 Retrieval 恢复入口"
            )
        if memory_enabled is not None:
            raise ValueError("首批 hypothesis operation 不接受 memory_enabled")
        base_input = {
            "scientific_question": scientific_question,
            "initial_chemistry_context": copy.deepcopy(dict(chemistry_context)),
            "generation_config": copy.deepcopy(dict(generation_config)),
            "stage_tool_projection": copy.deepcopy(list(normalized_tool_projection)),
        }
        provided_context = _project_calculation_unit_context(
            supplementary_scientific_input
        )
        if provided_context:
            base_input["initial_chemistry_context"] = copy.deepcopy(
                provided_context
            )
        if chemical_capability_projection:
            base_input["chemical_capability_projection"] = copy.deepcopy(
                list(chemical_capability_projection)
            )
        if has_frozen_retrieval:
            base_input["initial_retrieval_phase"] = copy.deepcopy(
                dict(initial_retrieval_phase or {})
            )
        else:
            base_input["retrieval_request"] = copy.deepcopy(
                dict(retrieval_request or {})
            )
        return base_input

    @staticmethod
    def _prepare_regeneration_hypothesis_operation_base_input(
        *,
        regeneration_package: Mapping[str, Any],
        generation_config: Mapping[str, Any],
        initial_query_angles: Optional[Sequence[str]],
        initial_retrieval_phase: Optional[Mapping[str, Any]],
        normalized_tool_projection: Sequence[Mapping[str, Any]],
        chemical_capability_projection: Sequence[Mapping[str, Any]] = (),
        memory_enabled: Optional[bool],
        retrieval_request: Optional[Mapping[str, Any]],
        retrieve: Optional[Callable[[Callable[[Mapping[str, Any]], None]], Any]],
    ) -> Dict[str, Any]:
        """校验再生成入口并构造不含来源编号的 base input。"""

        if not isinstance(memory_enabled, bool):
            raise ValueError("再生成必须显式说明 memory 是否启用")
        if initial_query_angles is None or not isinstance(
            initial_retrieval_phase, Mapping
        ):
            raise ValueError("再生成必须提供冻结 query angles 和 Retrieval")
        if retrieval_request is not None or retrieve is not None:
            raise ValueError("后续批不得重新运行 Retrieval")
        base_input = copy.deepcopy(dict(regeneration_package))
        base_input.pop("source_record_ids", None)
        base_input["generation_config"] = copy.deepcopy(dict(generation_config))
        base_input["stage_tool_projection"] = copy.deepcopy(
            list(normalized_tool_projection)
        )
        if chemical_capability_projection:
            base_input["chemical_capability_projection"] = copy.deepcopy(
                list(chemical_capability_projection)
            )
        return base_input

    @staticmethod
    def _prepare_hypothesis_operation_base_input(
        *,
        regeneration_package: Optional[Mapping[str, Any]],
        initial_query_angles: Optional[Sequence[str]],
        initial_retrieval_phase: Optional[Mapping[str, Any]],
        chemistry_context: Mapping[str, Any],
        scientific_question: str,
        generation_config: Mapping[str, Any],
        normalized_tool_projection: Sequence[Mapping[str, Any]],
        chemical_capability_projection: Sequence[Mapping[str, Any]] = (),
        memory_enabled: Optional[bool],
        retrieval_request: Optional[Mapping[str, Any]],
        retrieve: Optional[Callable[[Callable[[Mapping[str, Any]], None]], Any]],
        supplementary_scientific_input: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """校验首批或再生成入口并构造模型 operation 的 base input。"""

        if regeneration_package is None:
            return HypothesisAgent._prepare_initial_hypothesis_operation_base_input(
                initial_query_angles=initial_query_angles,
                initial_retrieval_phase=initial_retrieval_phase,
                chemistry_context=chemistry_context,
                scientific_question=scientific_question,
                generation_config=generation_config,
                normalized_tool_projection=normalized_tool_projection,
                chemical_capability_projection=chemical_capability_projection,
                memory_enabled=memory_enabled,
                retrieval_request=retrieval_request,
                retrieve=retrieve,
                supplementary_scientific_input=supplementary_scientific_input,
            )
        return HypothesisAgent._prepare_regeneration_hypothesis_operation_base_input(
            regeneration_package=regeneration_package,
            generation_config=generation_config,
            initial_query_angles=initial_query_angles,
            initial_retrieval_phase=initial_retrieval_phase,
            normalized_tool_projection=normalized_tool_projection,
            chemical_capability_projection=chemical_capability_projection,
            memory_enabled=memory_enabled,
            retrieval_request=retrieval_request,
            retrieve=retrieve,
        )

    @staticmethod
    def _resolve_hypothesis_operation_retrieval(
        *,
        runner: DynamicModelOperationRunner,
        regeneration_package: Optional[Mapping[str, Any]],
        initial_retrieval_phase: Optional[Mapping[str, Any]],
        retrieval_request: Optional[Mapping[str, Any]],
        retrieve: Optional[
            Callable[[Callable[[Mapping[str, Any]], None]], Any]
        ],
        chemistry_context: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """读取冻结 Retrieval 或恢复历史 Retrieval，并返回有效化学上下文。"""

        if regeneration_package is not None:
            return (
                copy.deepcopy(dict(initial_retrieval_phase or {})),
                copy.deepcopy(dict(chemistry_context)),
            )
        if isinstance(initial_retrieval_phase, Mapping):
            retrieval_result = copy.deepcopy(dict(initial_retrieval_phase))
        else:
            assert retrieval_request is not None and retrieve is not None
            try:
                retrieval_result = runner.run_retrieval_stage(
                    request=retrieval_request,
                    retrieve=retrieve,
                )
            except ModelOperationStageError as exc:
                raise HypothesisProtocolError(
                    "retrieval",
                    exc.failure_kind,
                ) from exc
        retrieved_context = retrieval_result.get("chemistry_context")
        effective_context = (
            copy.deepcopy(dict(retrieved_context))
            if isinstance(retrieved_context, Mapping)
            else copy.deepcopy(dict(chemistry_context))
        )
        return retrieval_result, effective_context

    def run_hypothesis_operation(
        self,
        *,
        task_root: str,
        operation_instance_id: str,
        scientific_question: str,
        stage_tool_projection: Sequence[Mapping[str, Any]],
        chemical_capability_projection: Sequence[Mapping[str, Any]] | None = None,
        initial_query_angles: Optional[Sequence[str]],
        initial_retrieval_phase: Optional[Mapping[str, Any]],
        chemistry_context: Mapping[str, Any],
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        hypothesis_ids_by_query: Sequence[Sequence[str]],
        merge_hypothesis_ids: Sequence[str],
        retrieval_request: Optional[Mapping[str, Any]] = None,
        retrieve: Optional[
            Callable[[Callable[[Mapping[str, Any]], None]], Any]
        ] = None,
        regeneration_package: Optional[Mapping[str, Any]] = None,
        memory_enabled: Optional[bool] = None,
        case_profile: str | None = None,
        prior_manifest_path: Optional[str] = None,
        observer: ModelStageObserver | None = None,
        display_context: Mapping[str, Any] | None = None,
        supplementary_scientific_input: Optional[Mapping[str, Any]] = None,
        expected_task_id: str | None = None,
        expected_batch_id: str | None = None,
        task_config_schema: Literal["legacy_workflow_v1", "investigation_v2"] = (
            "legacy_workflow_v1"
        ),
    ) -> Dict[str, Any]:
        """在一份动态清单中完成生成、比较、合并和候选排名。

        活动工作流在调用本方法前已经完成一次 Retrieval，并通过
        ``initial_retrieval_phase`` 传入冻结结果。旧清单恢复仍可传入
        ``retrieval_request`` 和 ``retrieve``，让尚未完成 Retrieval 的历史操作继续
        写完原清单；后续批只读取冻结检索和再生成历史。方法返回检索快照、规范假设
        结果和模型操作清单，含 Investigation Working Memory 时会核对显式
        expected task/batch 身份；模型协议失败时抛出 ``HypothesisProtocolError``。
        """

        self._validate_hypothesis_operation_schema(task_config_schema, top_n)
        if self._operation_runner is not None:
            raise RuntimeError("同一个 HypothesisAgent 不能同时运行两批假设")
        generation_config = self._build_hypothesis_operation_generation_config(
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            top_n=top_n,
            task_config_schema=task_config_schema,
        )
        normalized_tool_projection = (
            self._validate_stage_tool_projection(list(stage_tool_projection))
            if stage_tool_projection
            else []
        )
        normalized_capability_projection = (
            copy.deepcopy(list(chemical_capability_projection))
            if chemical_capability_projection
            else []
        )
        base_input = self._prepare_hypothesis_operation_base_input(
            regeneration_package=regeneration_package,
            initial_query_angles=initial_query_angles,
            initial_retrieval_phase=initial_retrieval_phase,
            chemistry_context=chemistry_context,
            scientific_question=scientific_question,
            generation_config=generation_config,
            normalized_tool_projection=normalized_tool_projection,
            chemical_capability_projection=normalized_capability_projection,
            memory_enabled=memory_enabled,
            retrieval_request=retrieval_request,
            retrieve=retrieve,
            supplementary_scientific_input=supplementary_scientific_input,
        )

        runner = DynamicModelOperationRunner(
            task_root=task_root,
            operation_instance_id=operation_instance_id,
            operation="hypothesis_generation",
            base_input=base_input,
            provider=self._call_model_provider,
            memory_enabled=memory_enabled,
            prior_manifest_path=prior_manifest_path,
            retry_failed_operation=prior_manifest_path is not None,
            observer=observer,
            display_context=display_context,
        )
        self._operation_runner = runner
        try:
            retrieval_result, effective_context = (
                self._resolve_hypothesis_operation_retrieval(
                    runner=runner,
                    regeneration_package=regeneration_package,
                    initial_retrieval_phase=initial_retrieval_phase,
                    retrieval_request=retrieval_request,
                    retrieve=retrieve,
                    chemistry_context=chemistry_context,
                )
            )

            generated = self.generate_hypothesis_batch(
                scientific_question=scientific_question,
                stage_tool_projection=normalized_tool_projection,
                chemical_capability_projection=normalized_capability_projection,
                initial_query_angles=initial_query_angles,
                initial_retrieval_phase=retrieval_result,
                chemistry_context=effective_context,
                num_queries=num_queries,
                num_hypotheses_per_query=num_hypotheses_per_query,
                top_n=top_n,
                hypothesis_ids_by_query=hypothesis_ids_by_query,
                merge_hypothesis_ids=merge_hypothesis_ids,
                regeneration_package=regeneration_package,
                memory_enabled=memory_enabled,
                case_profile=case_profile,
                supplementary_scientific_input=supplementary_scientific_input,
                expected_task_id=expected_task_id,
                expected_batch_id=expected_batch_id,
                task_config_schema=task_config_schema,
            )
            operation_result = runner.finalize_model_operation()
            if operation_result["status"] != "validated":
                raise RuntimeError("假设正文已返回，但 operation manifest 不是成功终态")
            return {
                "retrieval_result": retrieval_result,
                "hypothesis_result": generated,
                "operation_result": operation_result,
            }
        except HypothesisProtocolError as exc:
            operation_result = runner.finalize_model_operation()
            exc.operation_result = operation_result
            exc.manifest_ref = copy.deepcopy(operation_result["manifest_ref"])
            exc.manifest_path = operation_result["manifest_path"]
            raise
        finally:
            self._operation_runner = None

    @staticmethod
    def _build_hypothesis_replay_query_stage(
        *,
        num_queries: int,
        query_domain_validator: Callable[[Any], Any] | None,
    ) -> ModelStage:
        """构造首批 query stage 的固定校验规格。"""

        return ModelStage(
            stage="query",
            stage_kind="model",
            validation_kind="query_delimited_text",
            expected_count=num_queries,
            domain_validator=query_domain_validator,
            domain_validator_id="hypothesis.query_angles.v1",
        )

    @staticmethod
    def _build_hypothesis_replay_hypothesis_stage(
        *,
        stage_name: str,
        query_index: Any,
        num_hypotheses_per_query: int,
        hypothesis_domain_validator: Callable[[Any], Any],
    ) -> ModelStage:
        """构造首批 hypothesis 或后续 regeneration stage 的校验规格。"""

        return ModelStage(
            stage=stage_name,
            stage_kind="model",
            validation_kind="json_schema",
            query_index=query_index,
            schema=_hypothesis_array_schema(num_hypotheses_per_query),
            domain_validator=hypothesis_domain_validator,
            domain_validator_id="hypothesis.generated_single_claim.v4",
        )

    @staticmethod
    def _build_hypothesis_replay_relationship_stage(
        *,
        query_index: Any,
    ) -> ModelStage:
        """构造 relationship stage 的固定校验规格。"""

        return ModelStage(
            stage="relationship",
            stage_kind="model",
            validation_kind="json_schema",
            query_index=query_index,
            schema=_RELATIONSHIP_SCHEMA,
            domain_validator=_validate_relationship_payload,
            domain_validator_id="hypothesis.relationship_reasoned.v1",
        )

    @staticmethod
    def _build_hypothesis_replay_synthesis_stage(
        *,
        stage_name: str,
        query_index: Any,
        synthesis_domain_validator: Callable[[Any], Any],
    ) -> ModelStage:
        """构造 merge、complement 或 conflict stage 的校验规格。"""

        return ModelStage(
            stage=stage_name,
            stage_kind="model",
            validation_kind="json_schema",
            query_index=query_index,
            schema=_MERGED_HYPOTHESIS_SCHEMA,
            domain_validator=synthesis_domain_validator,
            domain_validator_id=f"hypothesis.{stage_name}_two_field.v1",
        )

    @staticmethod
    def _build_hypothesis_replay_rank_stage(
        *,
        query_index: Any,
    ) -> ModelStage:
        """构造 rank stage 的固定校验规格。"""

        return ModelStage(
            stage="rank",
            stage_kind="model",
            validation_kind="json_schema",
            query_index=query_index,
            schema=_RANK_SCHEMA,
            domain_validator=_validate_rank_payload,
            domain_validator_id="hypothesis.rank_reasoned.v1",
        )

    @staticmethod
    def _build_hypothesis_replay_phase_stage(
        *,
        stage_name: Any,
        query_index: Any,
        initial_operation: bool,
        num_queries: int,
        num_hypotheses_per_query: int,
        hypothesis_domain_validator: Callable[[Any], Any],
        query_domain_validator: Callable[[Any], Any] | None,
    ) -> ModelStage | None:
        """构造首批 query/hypothesis 或后续 regeneration stage。"""

        if initial_operation and stage_name == "query":
            return HypothesisAgent._build_hypothesis_replay_query_stage(
                num_queries=num_queries,
                query_domain_validator=query_domain_validator,
            )
        if initial_operation and stage_name == "hypothesis":
            return HypothesisAgent._build_hypothesis_replay_hypothesis_stage(
                stage_name=stage_name,
                query_index=query_index,
                num_hypotheses_per_query=num_hypotheses_per_query,
                hypothesis_domain_validator=hypothesis_domain_validator,
            )
        if not initial_operation and stage_name == "regeneration":
            return HypothesisAgent._build_hypothesis_replay_hypothesis_stage(
                stage_name=stage_name,
                query_index=query_index,
                num_hypotheses_per_query=num_hypotheses_per_query,
                hypothesis_domain_validator=hypothesis_domain_validator,
            )
        return None

    @staticmethod
    def _build_hypothesis_replay_synthesis_or_rank_stage(
        *,
        stage_name: Any,
        query_index: Any,
        synthesis_domain_validator: Callable[[Any], Any],
    ) -> ModelStage | None:
        """构造 relationship、合并类或 rank stage。"""

        if stage_name == "relationship":
            return HypothesisAgent._build_hypothesis_replay_relationship_stage(
                query_index=query_index,
            )
        if stage_name in {"merge", "complement", "conflict"}:
            return HypothesisAgent._build_hypothesis_replay_synthesis_stage(
                stage_name=stage_name,
                query_index=query_index,
                synthesis_domain_validator=synthesis_domain_validator,
            )
        if stage_name == "rank":
            return HypothesisAgent._build_hypothesis_replay_rank_stage(
                query_index=query_index,
            )
        return None

    @staticmethod
    def _build_hypothesis_replay_stage(
        raw_stage: Mapping[str, Any],
        *,
        stage_index: int,
        initial_operation: bool,
        num_queries: int,
        num_hypotheses_per_query: int,
        hypothesis_domain_validator: Callable[[Any], Any],
        synthesis_domain_validator: Callable[[Any], Any],
        query_domain_validator: Callable[[Any], Any] | None,
    ) -> ModelStage:
        """把 finalized 清单中的一个 stage 转成固定校验规格。"""

        from chemistry_multiagent.utils.model_operation import ModelOperationError

        if not isinstance(raw_stage, Mapping):
            prefix = "首批 " if initial_operation else ""
            raise ModelOperationError(
                f"{prefix}finalized stage {stage_index} 必须是 object"
            )
        if raw_stage.get("stage_kind") != "model":
            prefix = "首批 " if initial_operation else ""
            raise ModelOperationError(
                f"{prefix}finalized replay 只允许 model stage"
            )
        stage_name = raw_stage.get("stage")
        query_index = raw_stage.get("query_index")
        phase_stage = HypothesisAgent._build_hypothesis_replay_phase_stage(
            stage_name=stage_name,
            query_index=query_index,
            initial_operation=initial_operation,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            hypothesis_domain_validator=hypothesis_domain_validator,
            query_domain_validator=query_domain_validator,
        )
        if phase_stage is not None:
            return phase_stage
        synthesis_stage = HypothesisAgent._build_hypothesis_replay_synthesis_or_rank_stage(
            stage_name=stage_name,
            query_index=query_index,
            synthesis_domain_validator=synthesis_domain_validator,
        )
        if synthesis_stage is not None:
            return synthesis_stage
        prefix = "首批 " if initial_operation else ""
        raise ModelOperationError(
            f"{prefix}finalized hypothesis stage 未注册: {stage_name}"
        )

    @staticmethod
    def _require_hypothesis_replay_stage_list(
        raw_stages: Any,
        *,
        initial_operation: bool,
    ) -> list[Any]:
        """确认 finalized manifest 提供非空 stage 数组。"""

        from chemistry_multiagent.utils.model_operation import ModelOperationError

        if not isinstance(raw_stages, list) or not raw_stages:
            prefix = "首批 " if initial_operation else ""
            raise ModelOperationError(
                f"{prefix}hypothesis finalized manifest 缺少 stages"
            )
        return raw_stages

    @classmethod
    def _build_hypothesis_replay_domain_validators(
        cls,
        *,
        initial_operation: bool,
        num_queries: int,
        chemistry_context: Mapping[str, Any],
    ) -> tuple[
        Callable[[Any], Any],
        Callable[[Any], Any],
        Callable[[Any], Any] | None,
    ]:
        """按计算单元构造 hypothesis、合并和 query 的领域校验器。"""

        calculation_context = _project_calculation_unit_context(chemistry_context)
        calculation_unit = calculation_context.get("calculation_unit")
        has_calculation_unit = isinstance(calculation_unit, Mapping) and bool(
            calculation_unit
        )
        hypothesis_domain_validator = (
            _CalculationUnitHypothesisBatchValidator(calculation_unit)
            if has_calculation_unit
            else lambda value: cls._normalize_generated_hypotheses(value, [])
        )
        synthesis_domain_validator = (
            _CalculationUnitSynthesisValidator(calculation_unit)
            if has_calculation_unit
            else _validate_synthesis_payload
        )
        query_domain_validator = None
        if initial_operation:
            query_domain_validator = lambda value: cls._validate_query_angles(
                value,
                num_queries,
                calculation_unit=calculation_unit
                if isinstance(calculation_unit, Mapping)
                else None,
            )
        return (
            hypothesis_domain_validator,
            synthesis_domain_validator,
            query_domain_validator,
        )

    @classmethod
    def _build_hypothesis_replay_stage_specs(
        cls,
        raw_stages: Sequence[Any],
        *,
        initial_operation: bool,
        num_queries: int,
        num_hypotheses_per_query: int,
        domain_validators: tuple[
            Callable[[Any], Any],
            Callable[[Any], Any],
            Callable[[Any], Any] | None,
        ],
    ) -> list[ModelStage]:
        """把 finalized stage 数组逐项转换为固定校验规格。"""

        hypothesis_domain_validator, synthesis_domain_validator, query_domain_validator = (
            domain_validators
        )
        return [
            cls._build_hypothesis_replay_stage(
                raw_stage,
                stage_index=index,
                initial_operation=initial_operation,
                num_queries=num_queries,
                num_hypotheses_per_query=num_hypotheses_per_query,
                hypothesis_domain_validator=hypothesis_domain_validator,
                synthesis_domain_validator=synthesis_domain_validator,
                query_domain_validator=query_domain_validator,
            )
            for index, raw_stage in enumerate(raw_stages, start=1)
        ]

    @classmethod
    def _build_hypothesis_replay_stages(
        cls,
        raw_stages: Any,
        *,
        initial_operation: bool,
        num_queries: int,
        num_hypotheses_per_query: int,
        chemistry_context: Mapping[str, Any],
    ) -> list[ModelStage]:
        """校验 finalized stage 列表并构造全部重放校验规格。"""

        stages = cls._require_hypothesis_replay_stage_list(
            raw_stages,
            initial_operation=initial_operation,
        )
        domain_validators = cls._build_hypothesis_replay_domain_validators(
            initial_operation=initial_operation,
            num_queries=num_queries,
            chemistry_context=chemistry_context,
        )
        return cls._build_hypothesis_replay_stage_specs(
            stages,
            initial_operation=initial_operation,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            domain_validators=domain_validators,
        )

    def _execute_hypothesis_operation_replay(
        self,
        *,
        located: Mapping[str, Any],
        replay_stages: Sequence[ModelStage],
        task_root: str | Path,
        operation_instance_id: str,
        base_input: Mapping[str, Any],
        scientific_question: str,
        initial_query_angles: Sequence[str],
        initial_retrieval_phase: Mapping[str, Any],
        chemistry_context: Mapping[str, Any],
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        hypothesis_ids_by_query: Sequence[Sequence[str]],
        merge_hypothesis_ids: Sequence[str],
        regeneration_package: Mapping[str, Any],
        memory_enabled: bool,
        case_profile: str | None,
        supplementary_scientific_input: Mapping[str, Any] | None,
        expected_task_id: str | None,
        expected_batch_id: str | None,
        task_config_schema: Literal["legacy_workflow_v1", "investigation_v2"],
    ) -> Dict[str, Any]:
        """消费 finalized 后的后续批 stage，并返回确定性重放结果。"""

        replayed = ModelOperationRunner.replay_finalized_manifest(
            task_root=task_root,
            manifest_path=located["manifest_path"],
            operation="hypothesis_generation",
            operation_instance_id=operation_instance_id,
            expected_base_input=base_input,
            stages=replay_stages,
        )
        replay_runner = _FinalizedHypothesisReplayRunner(
            replay_stages,
            replayed["values"],
        )
        self._operation_runner = replay_runner  # type: ignore[assignment]
        try:
            generated = self.generate_hypothesis_batch(
                scientific_question=scientific_question,
                stage_tool_projection=base_input.get("stage_tool_projection", []),
                chemical_capability_projection=base_input.get(
                    "chemical_capability_projection", []
                ),
                initial_query_angles=list(initial_query_angles),
                initial_retrieval_phase=copy.deepcopy(dict(initial_retrieval_phase)),
                chemistry_context=copy.deepcopy(dict(chemistry_context)),
                num_queries=num_queries,
                num_hypotheses_per_query=num_hypotheses_per_query,
                top_n=top_n,
                hypothesis_ids_by_query=[list(group) for group in hypothesis_ids_by_query],
                merge_hypothesis_ids=list(merge_hypothesis_ids),
                regeneration_package=copy.deepcopy(dict(regeneration_package)),
                memory_enabled=memory_enabled,
                case_profile=case_profile,
                supplementary_scientific_input=(
                    copy.deepcopy(dict(supplementary_scientific_input))
                    if isinstance(supplementary_scientific_input, Mapping)
                    else None
                ),
                expected_task_id=expected_task_id,
                expected_batch_id=expected_batch_id,
                task_config_schema=task_config_schema,
            )
            replay_runner.assert_complete()
        finally:
            self._operation_runner = None
        return {
            "retrieval_result": copy.deepcopy(dict(initial_retrieval_phase)),
            "hypothesis_result": generated,
            "operation_result": {
                "status": "validated",
                "values": copy.deepcopy(replayed["values"]),
                "manifest": copy.deepcopy(replayed["manifest"]),
                "manifest_path": replayed["manifest_path"],
                "manifest_ref": copy.deepcopy(replayed["manifest_ref"]),
            },
        }

    def replay_hypothesis_operation_from_finalized_manifest(
        self,
        *,
        task_root: str | Path,
        operation_instance_id: str,
        base_input: Mapping[str, Any],
        scientific_question: str,
        initial_query_angles: Sequence[str],
        initial_retrieval_phase: Mapping[str, Any],
        chemistry_context: Mapping[str, Any],
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        hypothesis_ids_by_query: Sequence[Sequence[str]],
        merge_hypothesis_ids: Sequence[str],
        regeneration_package: Mapping[str, Any],
        memory_enabled: bool,
        case_profile: str | None = None,
        supplementary_scientific_input: Mapping[str, Any] | None = None,
        expected_task_id: str | None = None,
        expected_batch_id: str | None = None,
        task_config_schema: Literal["legacy_workflow_v1", "investigation_v2"] = (
            "legacy_workflow_v1"
        ),
    ) -> Dict[str, Any] | None:
        """严格重放已完成 Hypothesis operation，并重建同一最终 hypothesis_result。

        方法只读取固定 operation 目录中的 finalized manifest，先按当前 schema/domain
        validator 复核全部请求和回答，再把已校验值交给原有确定性归并/排名逻辑；不会
        调用 provider、重新发送模型请求或扫描其它 operation 目录。清单尚未 finalized
        时返回 ``None``，清单身份、base input、阶段顺序、回答或 expected task/batch
        身份不一致时抛出异常。
        """

        self._validate_hypothesis_operation_schema(task_config_schema, top_n)
        if self._operation_runner is not None:
            raise RuntimeError("同一个 HypothesisAgent 不能同时恢复两批假设")
        if not isinstance(base_input, Mapping):
            raise ValueError("replay base_input 必须是 object")
        if not isinstance(regeneration_package, Mapping):
            raise ValueError("replay regeneration_package 必须是 object")
        if not isinstance(memory_enabled, bool) or not memory_enabled:
            raise ValueError("Hypothesis operation finalized replay 只接受 Memory On")
        base_input = self._normalize_hypothesis_replay_base_input(
            base_input,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            top_n=top_n,
            task_config_schema=task_config_schema,
        )

        located = DynamicModelOperationRunner.locate_finalized_manifest_in_operation(
            task_root=task_root,
            operation="hypothesis_generation",
            operation_instance_id=operation_instance_id,
            expected_base_input=base_input,
        )
        if located is None:
            return None
        replay_stages = self._build_hypothesis_replay_stages(
            located["manifest"].get("stages"),
            initial_operation=False,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            chemistry_context=chemistry_context,
        )
        return self._execute_hypothesis_operation_replay(
            located=located,
            replay_stages=replay_stages,
            task_root=task_root,
            operation_instance_id=operation_instance_id,
            base_input=base_input,
            scientific_question=scientific_question,
            initial_query_angles=initial_query_angles,
            initial_retrieval_phase=initial_retrieval_phase,
            chemistry_context=chemistry_context,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            top_n=top_n,
            hypothesis_ids_by_query=hypothesis_ids_by_query,
            merge_hypothesis_ids=merge_hypothesis_ids,
            regeneration_package=regeneration_package,
            memory_enabled=memory_enabled,
            case_profile=case_profile,
            supplementary_scientific_input=supplementary_scientific_input,
            expected_task_id=expected_task_id,
            expected_batch_id=expected_batch_id,
            task_config_schema=task_config_schema,
        )

    def replay_initial_hypothesis_operation_from_finalized_manifest(
        self,
        *,
        task_root: str | Path,
        operation_instance_id: str,
        base_input: Mapping[str, Any],
        scientific_question: str,
        initial_retrieval_phase: Mapping[str, Any],
        chemistry_context: Mapping[str, Any],
        num_queries: int,
        num_hypotheses_per_query: int,
        top_n: int,
        hypothesis_ids_by_query: Sequence[Sequence[str]],
        merge_hypothesis_ids: Sequence[str],
        case_profile: str | None = None,
        supplementary_scientific_input: Mapping[str, Any] | None = None,
        task_config_schema: Literal["legacy_workflow_v1", "investigation_v2"] = (
            "legacy_workflow_v1"
        ),
    ) -> Dict[str, Any] | None:
        """严格重放首批 finalized manifest，不重新调用 Query 或 Hypothesis provider。"""

        self._validate_hypothesis_operation_schema(task_config_schema, top_n)
        if self._operation_runner is not None:
            raise RuntimeError("同一个 HypothesisAgent 不能同时恢复两批假设")
        if not isinstance(base_input, Mapping):
            raise ValueError("首批 replay base_input 必须是 object")

        from chemistry_multiagent.utils.model_operation import ModelOperationError

        located = DynamicModelOperationRunner.locate_finalized_manifest_in_operation(
            task_root=task_root,
            operation="hypothesis_generation",
            operation_instance_id=operation_instance_id,
            expected_base_input=base_input,
        )
        if located is None:
            return None
        replay_stages = self._build_hypothesis_replay_stages(
            located["manifest"].get("stages"),
            initial_operation=True,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            chemistry_context=chemistry_context,
        )

        replayed = ModelOperationRunner.replay_finalized_manifest(
            task_root=task_root,
            manifest_path=located["manifest_path"],
            operation="hypothesis_generation",
            operation_instance_id=operation_instance_id,
            expected_base_input=base_input,
            stages=replay_stages,
        )
        replay_runner = _FinalizedHypothesisReplayRunner(
            replay_stages,
            replayed["values"],
        )
        self._operation_runner = replay_runner  # type: ignore[assignment]
        try:
            generated = self.generate_hypothesis_batch(
                scientific_question=scientific_question,
                stage_tool_projection=base_input.get("stage_tool_projection", []),
                chemical_capability_projection=base_input.get(
                    "chemical_capability_projection", []
                ),
                initial_query_angles=None,
                initial_retrieval_phase=copy.deepcopy(dict(initial_retrieval_phase)),
                chemistry_context=copy.deepcopy(dict(chemistry_context)),
                num_queries=num_queries,
                num_hypotheses_per_query=num_hypotheses_per_query,
                top_n=top_n,
                hypothesis_ids_by_query=[list(group) for group in hypothesis_ids_by_query],
                merge_hypothesis_ids=list(merge_hypothesis_ids),
                regeneration_package=None,
                memory_enabled=None,
                case_profile=case_profile,
                supplementary_scientific_input=(
                    copy.deepcopy(dict(supplementary_scientific_input))
                    if isinstance(supplementary_scientific_input, Mapping)
                    else None
                ),
            )
            replay_runner.assert_complete()
        finally:
            self._operation_runner = None
        return {
            "retrieval_result": copy.deepcopy(dict(initial_retrieval_phase)),
            "hypothesis_result": generated,
            "operation_result": {
                "status": "validated",
                "values": copy.deepcopy(replayed["values"]),
                "manifest": copy.deepcopy(replayed["manifest"]),
                "manifest_path": replayed["manifest_path"],
                "manifest_ref": copy.deepcopy(replayed["manifest_ref"]),
            },
        }


@dataclass(frozen=True)
class _RankedHypothesisBatchPreparation:
    """保存一次批次模型调用前已经冻结的输入事实。"""

    batch_index: int
    previous_batch_id: str | None
    retrieval_result: Mapping[str, Any]
    num_queries: int
    hypotheses_per_query: int
    task_config_schema: str
    top_n: int
    hypothesis_ids_by_query: list[list[str]]
    merge_hypothesis_ids: list[str]
    causal_source_ref: str | None
    prior_manifest_path: str | None
    scientific_question: str
    initial_query_angles: list[str] | None
    chemistry_context: Mapping[str, Any]
    memory_enabled: bool | None
    regeneration_package: Mapping[str, Any] | None


class RankedHypothesisGenerator:
    """生成一批完整排名，并把入选假设建立为可运行组合。

    Retrieval 不在本类执行。总控在进入批次循环前只运行一次 Retrieval，并通过
    ``WorkflowState`` 冻结结果；首批和后续批次都调用同一个公开方法。后续批次是否
    能看到历史失败轨迹，仅由状态实现构造的再生成包决定。
    """

    def __init__(self, hypothesis_agent: HypothesisAgent) -> None:
        """绑定负责模型调用、假设合并和排名的底层 Agent。"""

        if not isinstance(hypothesis_agent, HypothesisAgent):
            raise TypeError("hypothesis_agent 必须是 HypothesisAgent")
        self.hypothesis_agent = hypothesis_agent

    @staticmethod
    def _resolve_hypothesis_task_config_schema(
        task_config: Mapping[str, Any],
    ) -> tuple[str, str]:
        """从冻结 task config 的字段来源解析 schema 和 Top N 字段。"""

        has_v2 = "top_n_hypotheses" in task_config
        has_legacy = "top_n_workflows" in task_config
        if has_v2 == has_legacy:
            raise ValueError(
                "Hypothesis task config 必须恰好提供 top_n_hypotheses 或 top_n_workflows"
            )
        if has_v2:
            return "investigation_v2", "top_n_hypotheses"
        return "legacy_workflow_v1", "top_n_workflows"

    @staticmethod
    def _validate_ranked_batch_request(
        batch_index: int,
        previous_batch_id: str | None,
    ) -> None:
        """校验批次序号和上一批身份，拒绝跳过或伪造批次链。"""

        if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 1:
            raise ValueError("batch_index 必须是正整数")
        if batch_index == 1 and previous_batch_id is not None:
            raise ValueError("首批不得引用 previous_batch_id")
        if batch_index > 1 and (
            not isinstance(previous_batch_id, str) or not previous_batch_id
        ):
            raise ValueError("后续批必须引用直接上一批")

    @staticmethod
    def _load_frozen_retrieval(
        context: "WorkflowRunContext",
    ) -> Dict[str, Any]:
        """读取任务已经冻结的 Retrieval；缺失时不启动假设模型。"""

        retrieval_result = context.state.load_retrieval_context()
        if not isinstance(retrieval_result, Mapping):
            raise RuntimeError("生成假设前必须完成并冻结一次 Retrieval")
        return copy.deepcopy(dict(retrieval_result))

    def _read_ranked_batch_budget(
        self,
        context: "WorkflowRunContext",
    ) -> tuple[int, int, str, int, list[list[str]], list[str]]:
        """读取冻结预算，并预先分配初始与合并候选的身份。"""

        task_config = context.task_config
        num_queries = self._require_positive_config(task_config, "num_queries")
        hypotheses_per_query = self._require_positive_config(
            task_config,
            "num_hypotheses_per_query",
        )
        task_config_schema, top_n_key = self._resolve_hypothesis_task_config_schema(
            task_config
        )
        top_n = self._require_positive_config(task_config, top_n_key)
        hypothesis_ids_by_query = [
            [
                context.identity_allocator("hypothesis")
                for _ in range(hypotheses_per_query)
            ]
            for _ in range(num_queries)
        ]
        candidate_count = num_queries * hypotheses_per_query
        # complete-link merge 组最多需要 candidate_count - 1 次 synthesis；继续按
        # 全量 pair 数保守预分配身份，以兼容既有恢复清单，未使用身份不会写入结果。
        synthesis_budget = max(0, candidate_count * (candidate_count - 1) // 2)
        merge_hypothesis_ids = [
            context.identity_allocator("hypothesis")
            for _ in range(synthesis_budget)
        ]
        return (
            num_queries,
            hypotheses_per_query,
            task_config_schema,
            top_n,
            hypothesis_ids_by_query,
            merge_hypothesis_ids,
        )

    @staticmethod
    def _load_hypothesis_restart(
        context: "WorkflowRunContext",
    ) -> tuple[str | None, str | None]:
        """读取模型失败后的清单引用，保持恢复因果链不变。"""

        restart = context.state.load_hypothesis_restart()
        if restart is None:
            return None, None
        if not isinstance(restart, Mapping):
            raise RuntimeError("hypothesis restart 必须是 object")
        causal_source_ref = restart.get("restart_record_id")
        prior_manifest_ref = restart.get("prior_manifest_ref")
        if not isinstance(causal_source_ref, str) or not causal_source_ref:
            raise RuntimeError("hypothesis restart 缺少 restart_record_id")
        return causal_source_ref, str(
            safe_join_task_path(context.task_root, prior_manifest_ref)
        )

    @staticmethod
    def _read_regeneration_scientific_context(
        regeneration_package: Mapping[str, Any],
    ) -> tuple[str, Sequence[str], Mapping[str, Any], Mapping[str, Any]]:
        """读取再生成包中的问题、角度、冻结 Retrieval 和化学上下文。"""

        scientific_context = regeneration_package.get("scientific_context")
        if not isinstance(scientific_context, Mapping):
            raise RuntimeError("regeneration package 缺少 scientific_context")
        scientific_question = scientific_context.get("scientific_question")
        initial_query_angles = scientific_context.get("initial_query_angles")
        frozen_retrieval = scientific_context.get("initial_retrieval_phase")
        chemistry_context = scientific_context.get("chemistry_context")
        if (
            not isinstance(scientific_question, str)
            or not isinstance(initial_query_angles, Sequence)
            or isinstance(initial_query_angles, (str, bytes))
            or not isinstance(frozen_retrieval, Mapping)
            or not isinstance(chemistry_context, Mapping)
        ):
            raise RuntimeError("regeneration package 的冻结科学上下文不完整")
        return (
            scientific_question,
            list(initial_query_angles),
            frozen_retrieval,
            chemistry_context,
        )

    @staticmethod
    def _assert_regeneration_retrieval_matches(
        *,
        scientific_question: str,
        initial_query_angles: Sequence[str],
        frozen_retrieval: Mapping[str, Any],
        retrieval_result: Mapping[str, Any],
        chemistry_context: Mapping[str, Any],
    ) -> None:
        """确认后续批次仍复用首批 Retrieval，不接受历史漂移。"""

        expected_retrieval = project_hypothesis_regeneration_context(
            scientific_question=scientific_question,
            initial_query_angles=list(initial_query_angles),
            initial_retrieval_phase=retrieval_result,
            chemistry_context=chemistry_context,
        )["initial_retrieval_phase"]
        if dict(frozen_retrieval) != expected_retrieval:
            raise RuntimeError("后续批次读取的 Retrieval 与首批冻结结果不一致")

    def _prepare_ranked_batch_scientific_context(
        self,
        context: "WorkflowRunContext",
        *,
        batch_index: int,
        retrieval_result: Mapping[str, Any],
    ) -> tuple[
        str,
        list[str] | None,
        Mapping[str, Any],
        bool | None,
        Mapping[str, Any] | None,
    ]:
        """准备首批或再生成批次的科学上下文和 Working Memory 包。"""

        scientific_question = context.scientific_question
        chemistry_context = context.scientific_context.get("chemistry_context", {})
        if batch_index == 1:
            if not isinstance(chemistry_context, Mapping):
                raise RuntimeError("任务科学上下文缺少 chemistry_context")
            return scientific_question, None, chemistry_context, None, None

        regeneration_package = context.state.build_hypothesis_regeneration_package(
            batch_index
        )
        if not isinstance(regeneration_package, Mapping):
            raise RuntimeError("hypothesis regeneration package 必须是 object")
        (
            scientific_question,
            initial_query_angles,
            frozen_retrieval,
            chemistry_context,
        ) = self._read_regeneration_scientific_context(regeneration_package)
        self._assert_regeneration_retrieval_matches(
            scientific_question=scientific_question,
            initial_query_angles=initial_query_angles,
            frozen_retrieval=frozen_retrieval,
            retrieval_result=retrieval_result,
            chemistry_context=chemistry_context,
        )
        return (
            scientific_question,
            initial_query_angles,
            chemistry_context,
            bool(context.state.memory_enabled),
            copy.deepcopy(dict(regeneration_package)),
        )

    def _prepare_ranked_batch_inputs(
        self,
        context: "WorkflowRunContext",
        batch_index: int,
        previous_batch_id: str | None,
    ) -> _RankedHypothesisBatchPreparation:
        """按任务阶段顺序准备 Retrieval、预算、恢复身份和再生成输入。"""

        retrieval_result = self._load_frozen_retrieval(context)
        (
            num_queries,
            hypotheses_per_query,
            task_config_schema,
            top_n,
            hypothesis_ids_by_query,
            merge_hypothesis_ids,
        ) = self._read_ranked_batch_budget(context)
        causal_source_ref, prior_manifest_path = self._load_hypothesis_restart(context)
        (
            scientific_question,
            initial_query_angles,
            chemistry_context,
            memory_enabled,
            regeneration_package,
        ) = self._prepare_ranked_batch_scientific_context(
            context,
            batch_index=batch_index,
            retrieval_result=retrieval_result,
        )
        return _RankedHypothesisBatchPreparation(
            batch_index=batch_index,
            previous_batch_id=previous_batch_id,
            retrieval_result=retrieval_result,
            num_queries=num_queries,
            hypotheses_per_query=hypotheses_per_query,
            task_config_schema=task_config_schema,
            top_n=top_n,
            hypothesis_ids_by_query=hypothesis_ids_by_query,
            merge_hypothesis_ids=merge_hypothesis_ids,
            causal_source_ref=causal_source_ref,
            prior_manifest_path=prior_manifest_path,
            scientific_question=scientific_question,
            initial_query_angles=initial_query_angles,
            chemistry_context=chemistry_context,
            memory_enabled=memory_enabled,
            regeneration_package=regeneration_package,
        )

    @staticmethod
    def _copy_supplementary_scientific_input(
        value: Any,
    ) -> Mapping[str, Any] | None:
        """复制可选科学补充输入；缺失或非 object 时保持旧的空值语义。"""

        if not isinstance(value, Mapping):
            return None
        return copy.deepcopy(dict(value))

    def _invoke_ranked_hypothesis_operation(
        self,
        context: "WorkflowRunContext",
        preparation: _RankedHypothesisBatchPreparation,
        *,
        operation_id: str,
        observer: ModelRunStageObserver,
    ) -> Any:
        """提交一次正式 Hypothesis operation，集中维护完整模型输入合同。"""

        initial_query_angles = (
            copy.deepcopy(preparation.initial_query_angles)
            if preparation.initial_query_angles is not None
            else None
        )
        stage_tool_projection = project_minichem_stage_tool_projection(
            context.catalog_snapshot,
            gaussian_route_policy=context.gaussian_route_policy,
        )
        chemical_capability_projection = (
            project_minichem_chemical_capability_projection(
                context.catalog_snapshot,
            )
        )
        return self.hypothesis_agent.run_hypothesis_operation(
            task_root=str(context.task_root.resolve()),
            operation_instance_id=operation_id,
            scientific_question=preparation.scientific_question,
            stage_tool_projection=stage_tool_projection,
            chemical_capability_projection=chemical_capability_projection,
            initial_query_angles=initial_query_angles,
            initial_retrieval_phase=copy.deepcopy(
                dict(preparation.retrieval_result)
            ),
            chemistry_context=copy.deepcopy(dict(preparation.chemistry_context)),
            num_queries=preparation.num_queries,
            num_hypotheses_per_query=preparation.hypotheses_per_query,
            top_n=preparation.top_n,
            hypothesis_ids_by_query=preparation.hypothesis_ids_by_query,
            merge_hypothesis_ids=preparation.merge_hypothesis_ids,
            regeneration_package=(
                copy.deepcopy(dict(preparation.regeneration_package))
                if preparation.regeneration_package is not None
                else None
            ),
            memory_enabled=preparation.memory_enabled,
            case_profile=context.case_profile,
            prior_manifest_path=preparation.prior_manifest_path,
            observer=observer,
            display_context={"batch_index": preparation.batch_index},
            supplementary_scientific_input=self._copy_supplementary_scientific_input(
                context.supplementary_scientific_input
            ),
            task_config_schema=preparation.task_config_schema,
        )

    @staticmethod
    def _save_ranked_hypothesis_protocol_pause(
        context: "WorkflowRunContext",
        error: HypothesisProtocolError,
        *,
        operation_id: str,
        causal_source_ref: str | None,
    ) -> Mapping[str, Any]:
        """保存模型协议暂停；清单身份缺失时按旧语义直接失败。"""

        manifest_ref = getattr(error, "manifest_ref", None)
        relative_path = (
            manifest_ref.get("relative_path")
            if isinstance(manifest_ref, Mapping)
            else None
        )
        if not isinstance(relative_path, str) or "/" not in relative_path:
            raise RuntimeError("失败的 hypothesis operation 缺少正式清单") from error
        reason_code = (
            "model_service_failure"
            if error.failure_kind == "transport"
            else "model_protocol_failure"
        )
        return context.state.save_hypothesis_pause(
            {
                "operation": "hypothesis_generation",
                "operation_instance_id": operation_id,
                "reason_code": reason_code,
                "causal_source_ref": causal_source_ref,
            },
            [relative_path],
        )

    def _run_ranked_hypothesis_operation(
        self,
        context: "WorkflowRunContext",
        preparation: _RankedHypothesisBatchPreparation,
        *,
        operation_id: str,
        observer: ModelRunStageObserver,
    ) -> tuple[Any, Mapping[str, Any] | None]:
        """运行 Hypothesis operation，并把协议失败转换为可恢复暂停。"""

        try:
            generated = self._invoke_ranked_hypothesis_operation(
                context,
                preparation,
                operation_id=operation_id,
                observer=observer,
            )
        except HypothesisProtocolError as error:
            return None, self._save_ranked_hypothesis_protocol_pause(
                context,
                error,
                operation_id=operation_id,
                causal_source_ref=preparation.causal_source_ref,
            )
        return generated, None

    def _read_validated_ranked_hypothesis_result(
        self,
        generated: Any,
    ) -> tuple[str, dict[str, Any]]:
        """校验 operation 清单并规范化模型返回的 Hypothesis phase。"""

        if not isinstance(generated, Mapping):
            raise RuntimeError("hypothesis operation 必须返回 object")
        operation_result = generated.get("operation_result")
        manifest_ref = (
            operation_result.get("manifest_ref")
            if isinstance(operation_result, Mapping)
            else None
        )
        manifest_path = (
            manifest_ref.get("relative_path")
            if isinstance(manifest_ref, Mapping)
            else None
        )
        if (
            not isinstance(operation_result, Mapping)
            or operation_result.get("status") != "validated"
            or not isinstance(manifest_path, str)
            or "/" not in manifest_path
        ):
            raise RuntimeError("通过的 hypothesis operation 缺少正式清单")
        phase = self._normalize_hypothesis_phase(generated.get("hypothesis_result"))
        return manifest_path, phase

    @staticmethod
    def _build_ranked_hypothesis_batch_payload(
        context: "WorkflowRunContext",
        preparation: _RankedHypothesisBatchPreparation,
        *,
        operation_id: str,
        phase: Mapping[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """按首批或 Working Memory 再生成语义建立可保存的批次事实。"""

        if preparation.batch_index == 1:
            question_snapshot: Mapping[str, Any] | None = {
                "scientific_question": preparation.scientific_question,
                "initial_query_angles": copy.deepcopy(phase["queries"]),
                "initial_retrieval_phase": copy.deepcopy(
                    dict(preparation.retrieval_result)
                ),
            }
            generation_source = {
                "kind": "initial",
                "previous_batch_id": None,
                "source_record_ids": [],
            }
        else:
            assert preparation.regeneration_package is not None
            question_snapshot = None
            generation_source = {
                "kind": "working_memory_regeneration",
                "previous_batch_id": preparation.previous_batch_id,
                "source_record_ids": copy.deepcopy(
                    list(preparation.regeneration_package["source_record_ids"])
                ),
            }
        return {
            "batch_id": context.identity_allocator("batch"),
            "batch_index": preparation.batch_index,
            "operation_instance_id": operation_id,
            "question_snapshot": question_snapshot,
            "generation_source": generation_source,
            "hypothesis_phase": copy.deepcopy(dict(phase)),
            "candidates": candidates,
            "causal_source_ref": preparation.causal_source_ref,
        }

    def generate_ranked_hypothesis_batch(
        self,
        context: "WorkflowRunContext",
        batch_index: int,
        previous_batch_id: str | None,
    ) -> Mapping[str, Any]:
        """按阶段顺序生成、校验并保存一个假设批次。

        输入是共享任务上下文、从 1 开始的批次序号和直接上一批身份。公开入口只编排
        输入冻结、正式 operation、结果校验、批次保存和展示发布；模型协议失败会保存
        任务级技术暂停，身份、清单或批次内容非法时直接抛错。
        """

        self._validate_ranked_batch_request(batch_index, previous_batch_id)
        preparation = self._prepare_ranked_batch_inputs(
            context,
            batch_index,
            previous_batch_id,
        )
        observer = ModelRunStageObserver(
            context.writer,
            display_context={"batch_index": batch_index},
        )
        operation_id = context.identity_allocator("hypothesis-operation")
        generated, pause_result = self._run_ranked_hypothesis_operation(
            context,
            preparation,
            operation_id=operation_id,
            observer=observer,
        )
        if pause_result is not None:
            return pause_result
        manifest_path, phase = self._read_validated_ranked_hypothesis_result(generated)
        candidates = self._build_batch_candidates(
            phase["ranked_strategies"],
            top_n=preparation.top_n,
            identity_allocator=context.identity_allocator,
        )
        batch_payload = self._build_ranked_hypothesis_batch_payload(
            context,
            preparation,
            operation_id=operation_id,
            phase=phase,
            candidates=candidates,
        )
        saved_batch = context.state.save_ranked_hypothesis_batch(
            batch_payload,
            [manifest_path],
        )
        self._write_hypothesis_display_files(
            context,
            batch_payload=batch_payload,
            observer=observer,
        )
        return saved_batch

    def _write_hypothesis_display_files(
        self,
        context: "WorkflowRunContext",
        *,
        batch_payload: Mapping[str, Any],
        observer: ModelRunStageObserver,
    ) -> None:
        """在批次事实保存成功后写入确定性 Hypothesis 摘要和最终排名。"""

        writer = context.writer
        if writer is None:
            return
        try:
            stage_paths = observer.get_stage_paths()
            ModelRunViewPublisher(writer).publish_hypothesis_summary(
                batch_payload,
                source_files=stage_paths,
                merge_decisions=observer.get_hypothesis_merge_decisions(),
                write_ranking=True,
            )
        except Exception:
            observer.mark_model_output_degraded("hypothesis_summary_publish_failed")

    @staticmethod
    def _require_positive_config(config: Mapping[str, Any], field_name: str) -> int:
        """读取一个正整数任务预算；缺失或非法时拒绝启动模型调用。"""

        value = config.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"task_config.{field_name} 必须是正整数")
        return value

    @staticmethod
    def _require_hypothesis_phase_mapping(value: Any) -> Mapping[str, Any]:
        """确认 hypothesis phase 是 object。"""

        if not isinstance(value, Mapping):
            raise RuntimeError("hypothesis phase 必须是 object")
        return value

    @staticmethod
    def _read_hypothesis_phase_queries(value: Any) -> list[str]:
        """读取 hypothesis phase 的非空、唯一研究角度。"""

        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
        ):
            raise RuntimeError("hypothesis phase 缺少研究角度、候选分组或完整排名")
        if len(set(value)) != len(value):
            raise RuntimeError("hypothesis phase 的 queries 不能重复")
        return value

    @staticmethod
    def _read_hypothesis_phase_collections(
        groups: Any,
        ranked: Any,
        query_count: int,
    ) -> tuple[list[Any], list[Any]]:
        """读取 hypothesis phase 的候选分组和完整排名数组。"""

        if not isinstance(groups, list) or not isinstance(ranked, list) or not ranked:
            raise RuntimeError("hypothesis phase 缺少研究角度、候选分组或完整排名")
        if len(groups) != query_count:
            raise RuntimeError("hypotheses_by_query 数量与 queries 不一致")
        return groups, ranked

    @staticmethod
    def _read_hypothesis_phase_sections(
        value: Any,
    ) -> tuple[list[str], list[Any], list[Any]]:
        """读取并检查研究角度、候选分组和排名容器。"""

        phase = RankedHypothesisGenerator._require_hypothesis_phase_mapping(value)
        queries = RankedHypothesisGenerator._read_hypothesis_phase_queries(
            phase.get("queries")
        )
        groups, ranked = RankedHypothesisGenerator._read_hypothesis_phase_collections(
            phase.get("hypotheses_by_query"),
            phase.get("ranked_strategies"),
            len(queries),
        )
        return queries, groups, ranked

    @classmethod
    def _normalize_hypothesis_phase_groups(
        cls,
        queries: Sequence[str],
        groups: Sequence[Any],
    ) -> tuple[list[dict[str, Any]], set[str], set[str]]:
        """按研究角度投影候选分组并核对候选身份唯一性。"""

        normalized_groups = []
        grouped_hypothesis_ids: set[str] = set()
        grouped_queries: set[str] = set()
        for group_index, group in enumerate(groups):
            if not isinstance(group, Mapping) or group.get("query") != queries[group_index]:
                raise RuntimeError("hypotheses_by_query 必须逐项对应 queries")
            if queries[group_index] in grouped_queries:
                raise RuntimeError("hypotheses_by_query 的 source query 不能重复")
            grouped_queries.add(queries[group_index])
            hypotheses = group.get("hypotheses")
            if not isinstance(hypotheses, list) or not hypotheses:
                raise RuntimeError("hypotheses_by_query 分组缺少候选")
            projected_hypotheses = [cls._project_hypothesis(item) for item in hypotheses]
            for hypothesis in projected_hypotheses:
                hypothesis_id = hypothesis["hypothesis_id"]
                if hypothesis_id in grouped_hypothesis_ids:
                    raise RuntimeError("hypotheses_by_query 的 hypothesis_id 不能重复")
                grouped_hypothesis_ids.add(hypothesis_id)
            normalized_groups.append(
                {
                    "query": queries[group_index],
                    "hypotheses": projected_hypotheses,
                }
            )
        return normalized_groups, grouped_hypothesis_ids, grouped_queries

    @classmethod
    def _normalize_hypothesis_phase_ranked(
        cls,
        ranked: Sequence[Any],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """按连续排名投影候选并核对排名身份唯一性。"""

        normalized_ranked = [cls._project_hypothesis(item) for item in ranked]
        ranked_ids: set[str] = set()
        for rank, hypothesis in enumerate(normalized_ranked, start=1):
            if hypothesis["hypothesis_id"] in ranked_ids:
                raise RuntimeError("ranked_strategies 的 hypothesis_id 不能重复")
            ranked_ids.add(hypothesis["hypothesis_id"])
            hypothesis.setdefault("rank", rank)
            if hypothesis["rank"] != rank:
                raise RuntimeError("ranked strategy 的 rank 必须从 1 连续排列")
        return normalized_ranked, ranked_ids

    @staticmethod
    def _read_consumed_hypothesis_ids(
        ranked: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        """读取完整排名中由 synthesis_sources 消费的原始父项身份。"""

        return _read_consumed_hypothesis_ids(ranked)

    @classmethod
    def _normalize_hypothesis_phase(cls, value: Any) -> dict[str, Any]:
        """校验模型批次的研究角度、分组候选和完整排名。"""

        queries, groups, ranked = cls._read_hypothesis_phase_sections(value)
        normalized_groups, grouped_ids, grouped_queries = (
            cls._normalize_hypothesis_phase_groups(queries, groups)
        )
        normalized_ranked, ranked_ids = cls._normalize_hypothesis_phase_ranked(ranked)
        if grouped_queries != set(queries):
            raise RuntimeError("hypotheses_by_query 没有完整覆盖 queries")
        consumed_ids = cls._read_consumed_hypothesis_ids(normalized_ranked)
        if not grouped_ids <= ranked_ids | consumed_ids:
            raise RuntimeError("hypotheses_by_query 使用了不在完整排名中的 hypothesis_id")
        return {
            "queries": copy.deepcopy(queries),
            "hypotheses_by_query": normalized_groups,
            "ranked_strategies": normalized_ranked,
        }

    @staticmethod
    def _project_hypothesis(value: Any) -> dict[str, Any]:
        """保留规划和审计继续使用的规范假设字段。"""

        required = ("hypothesis_id", "strategy_name", "reasoning")
        if not isinstance(value, Mapping) or any(
            not isinstance(value.get(field), str) or not value[field].strip()
            for field in required
        ):
            raise RuntimeError("hypothesis 科学正文不完整")
        result = {field: value[field] for field in required}
        for field in (
            "merged_from",
            "synthesis_sources",
            "wins",
            "losses",
            "score",
            "rank",
        ):
            if field in value:
                result[field] = copy.deepcopy(value[field])
        return result

    @staticmethod
    def _build_batch_candidates(
        ranked: Sequence[Mapping[str, Any]],
        *,
        top_n: int,
        identity_allocator: Callable[[str], str],
    ) -> list[dict[str, Any]]:
        """为完整排名建立候选索引，并只给入选项分配组合身份。"""

        identities: set[str] = set()
        candidates = []
        selected_count = min(top_n, len(ranked))
        for candidate_index, hypothesis in enumerate(ranked, start=1):
            hypothesis_id = hypothesis.get("hypothesis_id")
            if not isinstance(hypothesis_id, str) or not hypothesis_id:
                raise RuntimeError("ranked hypothesis 缺少 hypothesis_id")
            if hypothesis_id in identities:
                raise RuntimeError("同一批 ranked hypotheses 不得复用 hypothesis_id")
            identities.add(hypothesis_id)
            selected = candidate_index <= selected_count
            candidates.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "candidate_index": candidate_index,
                    "rank": candidate_index,
                    "selection_status": "selected" if selected else "not_selected",
                    "combination_id": (
                        identity_allocator("combination") if selected else None
                    ),
                    "combination_index": candidate_index if selected else None,
                }
            )
        return candidates


__all__ = [
    "HypothesisAgent",
    "HypothesisProtocolError",
    "RankedHypothesisGenerator",
    "project_scientific_hypothesis_identity",
]
