"""Investigation and Experiment 阶段的 prompt/response 合同。

本模块只负责输入的结构化校验、响应模型和 JSON Schema，不启动模型、不调用
MiniChem，也不保存会话。提示构造器使用这里的确定性 JSON 规范化函数，因而同一
组事实总会产生同一份请求正文；非法路径、凭据和额外字段会在发送前失败。
"""

from __future__ import annotations

import copy
import ipaddress
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias, Union
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)


__all__ = [
    "InvestigationResponse",
    "PassReviewFinding",
    "FailReviewFinding",
    "TaskReviewResponse",
    "TaskFinalizationFinding",
    "TaskFinalizationResponse",
    "investigation_response_json_schema",
    "review_response_json_schema",
    "finalization_response_json_schema",
    "build_investigation_provider_json_schema",
    "build_task_review_provider_json_schema",
    "build_task_finalization_provider_json_schema",
    "validate_investigation_response",
    "validate_task_review_response",
    "validate_task_finalization_response",
    "require_safe_public_scientific_text",
    "require_safe_public_scientific_context",
    "require_safe_scientific_question_text",
]


# 这些词不是调查结果合同中的状态。它们放在拒绝清单中，避免模型自行发明
# 额外的中间路由或第三种科学判断。
_FORBIDDEN_STATUS_TERMS = frozenset(
    {
        "waiting",
        "waiting_for_jobs",
        "inconclusive",
        "continue_investigation",
    }
)

_JSON_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_PATH_OR_SECRET_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9_])/(?:tmp|mnt|home|root|var|etc|opt)(?:/|$)|"
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|"
    r"(?:^|[\s])(?:\\\\|\.\.?/)|"
    r"\bsk-[A-Za-z0-9_-]{8,}\b|"
    r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[^\s,;]+"
    r")",
    re.IGNORECASE,
)
_URI_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]{1,31}://[^\s<>\"']+",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_DISPLAY_TEXT_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PUBLIC_CONTEXT_STRICT_KEY_NAMES = frozenset(
    {
        "task",
        "taskid",
        "researchtaskid",
        "batch",
        "batchid",
        "investigationcontextid",
        "session",
        "sessionid",
        "threadid",
        "codexthreadid",
        "reviewthreadid",
        "reviewturnid",
        "turnid",
        "model",
        "modelname",
        "provider",
        "operation",
        "artifact",
        "artifacts",
        "artifactid",
        "artifactref",
        "artifactrefs",
        "sourceartifact",
        "sourceartifactref",
        "sourceartifactrefs",
        "operationid",
        "operationref",
        "operationrefs",
        "jobid",
        "jobref",
        "jobrefs",
        "scopeid",
        "workspaceref",
        "eventid",
        "recordid",
        "snapshotid",
        "hypothesisid",
        "combinationid",
        "hypothesisexecutioninputrecordid",
        "resultversion",
        "operationinstanceid",
        "operationkey",
        "job",
        "sourcerecordids",
    }
)
_PUBLIC_CONTEXT_FORBIDDEN_KEY_NAMES = frozenset(
    {
        "workspace",
        "workspaceref",
        "scopeid",
        "thread",
        "threadid",
        "codexthreadid",
        "reviewthreadid",
        "reviewturnid",
        "transcript",
        "workflow",
        "currentworkflow",
        "runtime",
        "runtimecheckpoint",
        "checkpoint",
        "path",
        "paths",
        "manifestpath",
        "resumestatepath",
        "sourceref",
        "sourcerefs",
        "eventid",
        "recordid",
        "apikey",
        "accesstoken",
        "secret",
        "password",
        "credential",
        "credentials",
        "rawoutput",
        "rawtooloutput",
        "stdout",
        "stderr",
        "tooloutput",
    }
)


class _StrictContract(BaseModel):
    """所有响应对象共用严格字段、冻结和赋值校验策略。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


def _require_non_empty_text(
    value: Any,
    label: str,
    *,
    allow_public_source_uris: bool = False,
) -> str:
    """校验公开文本，并按字段边界允许安全的科学来源 URI。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{label} 不能包含控制字符")
    if _PATH_OR_SECRET_RE.search(value):
        raise ValueError(f"{label} 不能包含本机路径、URI 或凭据")
    for match in _URI_RE.finditer(value):
        uri = match.group(0).rstrip(".,;:)]}")
        if not allow_public_source_uris or not _is_public_source_uri(uri):
            raise ValueError(f"{label} 不能包含本机路径、URI 或凭据")
    return value.strip()


def _require_safe_reference(value: Any, label: str) -> str:
    """校验一个作业、操作或产物的不透明引用。"""

    return _require_non_empty_text(value, label)


def _validate_reference_tuple(value: tuple[str, ...], label: str) -> tuple[str, ...]:
    """校验引用集合无重复且不携带本机路径。"""

    normalized = tuple(_require_safe_reference(item, label) for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} 不能包含重复引用")
    return normalized


def _validate_text_tuple(value: tuple[str, ...], label: str) -> tuple[str, ...]:
    """校验观测或限制文本集合，保留调用方顺序但去除首尾空白。"""

    normalized = tuple(_require_non_empty_text(item, label) for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} 不能包含重复文本")
    return normalized


def _coerce_json_array_to_tuple(value: Any, label: str) -> tuple[Any, ...]:
    """把 JSON array 转为冻结合同使用的 tuple，拒绝字符串和其它序列。"""

    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    raise ValueError(f"{label} 必须是 JSON array")


def _is_public_source_uri(value: str) -> bool:
    """判断一个 URI 是否是可放入科学来源上下文的公开 HTTP(S) 地址。"""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(
        (".local", ".internal")
    ):
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    ):
        return False
    return True


def _assert_no_path_or_secret(
    value: Any,
    *,
    label: str = "输入",
    allow_public_source_uris: bool = False,
    allow_display_whitespace: bool = False,
) -> None:
    """递归检查结构化输入，不允许路径、URI、控制字符或常见 API 凭据。

    输入可以是 JSON object、array、字符串、布尔值、整数、有限浮点数或 null。
    非 JSON 值、非字符串对象键和非有限数字都会失败；调用方若允许公开来源，
    可通过参数保留公开 HTTP(S) URI，但 file://、宿主地址和凭据仍会失败。
    ``allow_display_whitespace`` 只供科学问题正文的窄校验入口使用；它仅允许
    制表符、换行和回车，不会改变路径、URI 或凭据检测。
    """

    if isinstance(value, str):
        has_control_character = (
            _DISPLAY_TEXT_CONTROL_RE.search(value)
            if allow_display_whitespace
            else _CONTROL_RE.search(value)
        )
        if has_control_character or _PATH_OR_SECRET_RE.search(value):
            raise ValueError(f"{label} 不能包含本机路径、URI、控制字符或凭据")
        for match in _URI_RE.finditer(value):
            uri = match.group(0).rstrip(".,;:)]}")
            if not allow_public_source_uris or not _is_public_source_uri(uri):
                raise ValueError(f"{label} 不能包含本机路径、URI、控制字符或凭据")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} 不能包含 NaN 或无穷数字")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} 的 object key 必须是非空字符串")
            _assert_no_path_or_secret(
                key,
                label=f"{label}.<key>",
                allow_public_source_uris=allow_public_source_uris,
                allow_display_whitespace=allow_display_whitespace,
            )
            _assert_no_path_or_secret(
                item,
                label=f"{label}.{key}",
                allow_public_source_uris=allow_public_source_uris,
                allow_display_whitespace=allow_display_whitespace,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_no_path_or_secret(
                item,
                label=f"{label}[{index}]",
                allow_public_source_uris=allow_public_source_uris,
                allow_display_whitespace=allow_display_whitespace,
            )
        return
    raise TypeError(f"{label} 不是可序列化的 JSON 值")


def _normalize_context_key(value: str) -> str:
    """将科学上下文字段名归一化为安全边界比较用的标识。"""

    return re.sub(r"[^a-z0-9]", "", value.casefold())


def require_safe_public_scientific_text(
    value: Any,
    label: str = "scientific_text",
) -> str:
    """校验公开科学正文，允许排版空白但拒绝路径、URI 和凭据。

    该入口只供科学问题和公开科学上下文正文使用。普通标识、引用、artifact
    和模型响应文本仍走严格的单行校验；即使这里允许制表符、换行和回车，也继续
    拒绝其它控制字符、本机路径、file URI、私有宿主 URI 以及常见 credential。
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    if _DISPLAY_TEXT_CONTROL_RE.search(value):
        raise ValueError(f"{label} 不能包含本机路径、URI、控制字符或凭据")
    _assert_no_path_or_secret(
        value,
        label=label,
        allow_public_source_uris=True,
        allow_display_whitespace=True,
    )
    return value


def require_safe_scientific_question_text(
    value: Any,
    label: str = "scientific_question",
) -> str:
    """校验公开科学问题正文，复用安全公开科学正文边界。"""

    return require_safe_public_scientific_text(value, label=label)


def _copy_strict_public_context_value(value: Any, label: str) -> Any:
    """递归复制身份引用值，并对其全部嵌套字段使用严格单行边界。"""

    if isinstance(value, str):
        _assert_no_path_or_secret(value, label=label)
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        _assert_no_path_or_secret(value, label=label)
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} 的 object key 必须是非空字符串")
            normalized_key = _normalize_context_key(key)
            if normalized_key in _PUBLIC_CONTEXT_FORBIDDEN_KEY_NAMES:
                raise ValueError(f"{label}.{key} 不能包含私有或凭据字段")
            _assert_no_path_or_secret(key, label=f"{label}.<key>")
            copied[key] = _copy_strict_public_context_value(
                item,
                f"{label}.{key}",
            )
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _copy_strict_public_context_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{label} 不是可序列化的 JSON 值")


def require_safe_public_scientific_context(
    value: Any,
    label: str = "scientific_context",
) -> Any:
    """递归复制安全公开科学 JSON，并让身份引用继续使用严格单行校验。

    普通科学字符串值允许制表符、换行和回车；object key、任务/批次/session/thread
    身份、模型名以及 job/operation/artifact 引用仍拒绝排版控制字符、路径、私有 URI
    和凭据。返回值只包含独立的 dict/list、字符串、有限数字、布尔值和 null，不改变
    正文中的允许空白，也不负责删除 Working Memory 的私有字段。
    """

    if isinstance(value, str):
        return require_safe_public_scientific_text(value, label=label)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        _assert_no_path_or_secret(value, label=label)
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} 的 object key 必须是非空字符串")
            normalized_key = _normalize_context_key(key)
            if normalized_key in _PUBLIC_CONTEXT_FORBIDDEN_KEY_NAMES:
                raise ValueError(f"{label}.{key} 不能包含私有或凭据字段")
            _assert_no_path_or_secret(key, label=f"{label}.<key>")
            item_label = f"{label}.{key}"
            if normalized_key in _PUBLIC_CONTEXT_STRICT_KEY_NAMES:
                copied[key] = _copy_strict_public_context_value(item, item_label)
            else:
                copied[key] = require_safe_public_scientific_context(
                    item,
                    label=item_label,
                )
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            require_safe_public_scientific_context(
                item,
                label=f"{label}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{label} 不是可序列化的 JSON 值")


def _canonicalize_json_value(
    value: Any,
    *,
    label: str = "输入",
    allow_public_source_uris: bool = False,
) -> Any:
    """验证并复制 JSON 值，按稳定键顺序返回不含 tuple/set 的普通值。

    该函数是 prompt 构造器的唯一输入归一化入口。它使用严格 JSON 编码拒绝
    隐式对象、NaN 和无穷数，随后重新解析编码结果，确保插入消息的值与调用方
    无关且序列化结果确定。
    """

    _assert_no_path_or_secret(
        value,
        label=label,
        allow_public_source_uris=allow_public_source_uris,
    )
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 不能规范化为严格 JSON") from exc


def _copy_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """返回独立的 schema 副本，防止调用方修改共享合同。"""

    return copy.deepcopy(dict(schema))


class InvestigationResponse(_StrictContract):
    """一次调查 turn 的最终科学结果。

    ``narrative`` 是回顾性计算过程，不要求组成可冷启动执行的 workflow。三类
    引用必须来自实际核对过的 MiniChem 结果；本模型不接受中间等待或第三种判断。
    """

    self_judgment: Literal["supported", "refuted"]
    narrative: StrictStr
    job_refs: tuple[StrictStr, ...]
    operation_refs: tuple[StrictStr, ...]
    artifact_refs: tuple[StrictStr, ...]
    observations: tuple[StrictStr, ...]
    limitations: tuple[StrictStr, ...]

    @field_validator("narrative")
    @classmethod
    def validate_narrative(cls, value: str) -> str:
        return _require_non_empty_text(
            value,
            "narrative",
            allow_public_source_uris=True,
        )

    @field_validator("job_refs", "operation_refs", "artifact_refs", mode="before")
    @classmethod
    def coerce_reference_arrays(cls, value: Any, info: Any) -> tuple[Any, ...]:
        return _coerce_json_array_to_tuple(value, str(getattr(info, "field_name", "引用")))

    @field_validator("observations", "limitations", mode="before")
    @classmethod
    def coerce_text_arrays(cls, value: Any, info: Any) -> tuple[Any, ...]:
        return _coerce_json_array_to_tuple(value, str(getattr(info, "field_name", "文本")))

    @field_validator("job_refs", "operation_refs", "artifact_refs")
    @classmethod
    def validate_references(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        return _validate_reference_tuple(value, str(getattr(info, "field_name", "引用")))

    @field_validator("observations", "limitations")
    @classmethod
    def validate_texts(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        normalized = tuple(
            _require_non_empty_text(
                item,
                str(getattr(info, "field_name", "文本")),
                allow_public_source_uris=True,
            )
            for item in value
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{getattr(info, 'field_name', '文本')} 不能包含重复文本")
        return normalized

    @field_validator("observations")
    @classmethod
    def require_observation(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("observations 至少需要一项实际观测")
        return value


class PassReviewFinding(_StrictContract):
    """Review 对一条假设执行输入判定有效并给出科学结论。"""

    hypothesis_execution_input_record_id: StrictStr
    validity: Literal["pass"]
    hypothesis_judgment: Literal["supported", "refuted"]

    @field_validator("hypothesis_execution_input_record_id")
    @classmethod
    def validate_execution_input_id(cls, value: str) -> str:
        return _require_non_empty_text(value, "hypothesis_execution_input_record_id")


class FailReviewFinding(_StrictContract):
    """Review 对一条假设执行输入判定证据不足并给出返修建议。"""

    hypothesis_execution_input_record_id: StrictStr
    validity: Literal["fail"]
    repair_advice: StrictStr

    @field_validator("hypothesis_execution_input_record_id", "repair_advice")
    @classmethod
    def validate_finding_text(cls, value: str, info: Any) -> str:
        field_name = str(getattr(info, "field_name", "字段"))
        return _require_non_empty_text(
            value,
            field_name,
            allow_public_source_uris=field_name == "repair_advice",
        )


_ReviewFinding: TypeAlias = Annotated[
    Union[PassReviewFinding, FailReviewFinding],
    Field(discriminator="validity"),
]


def _classify_review_findings(value: Any) -> Literal["supported", "fail", "refuted"] | None:
    """从原始 finding 识别回答文本分支，供解析前屏障使用。

    这里只读取已知的 ``validity`` 和 ``hypothesis_judgment`` 字段；其它结构错误
    仍交给 Pydantic 的 finding 联合模型报告，避免在解析前重复实现完整校验。
    """

    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 5:
        return None
    validities: list[Any] = []
    judgments: list[Any] = []
    for item in value:
        if isinstance(item, Mapping):
            validities.append(item.get("validity"))
            judgments.append(item.get("hypothesis_judgment"))
        elif isinstance(item, (PassReviewFinding, FailReviewFinding)):
            validities.append(item.validity)
            judgments.append(getattr(item, "hypothesis_judgment", None))
        else:
            return None
    if "fail" in validities:
        return "fail"
    if all(item == "pass" for item in validities):
        if "supported" in judgments:
            return "supported"
        if all(item == "refuted" for item in judgments):
            return "refuted"
    return None


class TaskReviewResponse(_StrictContract):
    """一次任务级 Review 的完整当前批次判断。

    Review 必须一次覆盖一至五个当前结果；精确批次数量由冻结任务模式约束。有效性为 ``pass`` 时只允许
    ``supported`` 或 ``refuted``；有效性为 ``fail`` 时只允许 ``repair_advice``。
    ``can_answer_original_question`` 为 true 时必须保存可直接返回用户的最终回答，
    且全部 validity 通过、至少一项为 supported；只有全部 validity 通过、
    至少一项为 supported 但仍不能回答原问题时，false 分支才保存回答能力说明。
    任一 validity 失败或全部 refuted 时，false 分支只能保留各项自己的
    ``repair_advice``（或由领域状态进入 revise_hypothesis），不得携带回答文本。
    """

    findings: tuple[_ReviewFinding, ...]
    can_answer_original_question: StrictBool
    final_answer: StrictStr | None = None
    answerability_explanation: StrictStr | None = None

    @field_validator("findings", mode="before")
    @classmethod
    def coerce_findings_array(cls, value: Any) -> tuple[Any, ...]:
        return _coerce_json_array_to_tuple(value, "findings")

    @model_validator(mode="before")
    @classmethod
    def validate_explanation_presence(cls, value: Any) -> Any:
        """按当前全部 finding 的类型在解析前限制回答文本的字段存在性。"""

        if not isinstance(value, Mapping):
            return value
        answer = value.get("can_answer_original_question")
        has_explanation = "answerability_explanation" in value
        explanation = value.get("answerability_explanation")
        has_final_answer = "final_answer" in value
        final_answer = value.get("final_answer")
        if answer is True and has_explanation:
            raise ValueError(
                "can_answer_original_question=true 不能携带 explanation"
            )
        if answer is True and (not has_final_answer or final_answer is None):
            raise ValueError(
                "can_answer_original_question=true 必须携带 final_answer"
            )
        if answer is False:
            finding_kind = _classify_review_findings(value.get("findings"))
            if finding_kind == "supported":
                if not has_explanation or explanation is None:
                    raise ValueError(
                        "全部 validity 为 pass 且存在 supported 时，"
                        "can_answer_original_question=false 必须携带 explanation"
                    )
                if has_final_answer:
                    raise ValueError(
                        "can_answer_original_question=false 不能携带 final_answer"
                    )
            elif finding_kind in {"fail", "refuted"}:
                if has_final_answer or has_explanation:
                    raise ValueError(
                        "任一 validity=fail 或全部 refuted 时，"
                        "不能携带 final_answer 或 answerability_explanation"
                    )
        return value

    @field_validator("findings")
    @classmethod
    def validate_current_findings(
        cls,
        value: tuple[_ReviewFinding, ...],
    ) -> tuple[_ReviewFinding, ...]:
        if not 1 <= len(value) <= 5:
            raise ValueError("Review 必须包含一至五项当前结果判断")
        execution_input_ids = tuple(
            item.hypothesis_execution_input_record_id for item in value
        )
        if len(set(execution_input_ids)) != len(execution_input_ids):
            raise ValueError("Review 的 hypothesis_execution_input_record_id 必须唯一")
        return value

    @field_validator("final_answer", "answerability_explanation")
    @classmethod
    def validate_answerability_text(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _require_non_empty_text(
            value,
            str(getattr(info, "field_name", "回答文本")),
            allow_public_source_uris=True,
        )

    @model_validator(mode="after")
    def validate_answerability_contract(self) -> "TaskReviewResponse":
        if self.can_answer_original_question:
            if self.final_answer is None:
                raise ValueError(
                    "can_answer_original_question=true 必须携带 final_answer"
                )
            if self.answerability_explanation is not None:
                raise ValueError(
                    "can_answer_original_question=true 不能携带 explanation"
                )
            if any(
                isinstance(finding, FailReviewFinding) for finding in self.findings
            ):
                raise ValueError(
                    "can_answer_original_question=true 时全部调查有效性必须全部为 pass"
                )
            if not any(
                isinstance(finding, PassReviewFinding)
                and finding.hypothesis_judgment == "supported"
                for finding in self.findings
            ):
                raise ValueError(
                    "can_answer_original_question=true 时至少一项判断必须为 supported"
                )
        else:
            has_fail = any(
                isinstance(finding, FailReviewFinding) for finding in self.findings
            )
            has_supported = any(
                isinstance(finding, PassReviewFinding)
                and finding.hypothesis_judgment == "supported"
                for finding in self.findings
            )
            if not has_fail and has_supported:
                if self.answerability_explanation is None:
                    raise ValueError(
                        "全部 validity 为 pass 且存在 supported 时，"
                        "can_answer_original_question=false 必须携带 explanation"
                    )
                if self.final_answer is not None:
                    raise ValueError(
                        "can_answer_original_question=false 不能携带 final_answer"
                    )
            elif self.final_answer is not None or self.answerability_explanation is not None:
                raise ValueError(
                    "任一 validity=fail 或全部 refuted 时，"
                    "不能携带 final_answer 或 answerability_explanation"
                )
        return self


class TaskFinalizationFinding(_StrictContract):
    """返修预算耗尽后，终局成稿对一条假设执行输入的证据摘要。"""

    hypothesis_execution_input_record_id: StrictStr
    summary: StrictStr

    @field_validator("hypothesis_execution_input_record_id", "summary")
    @classmethod
    def validate_finalization_text(cls, value: str, info: Any) -> str:
        field_name = str(getattr(info, "field_name", "字段"))
        return _require_non_empty_text(
            value,
            field_name,
            allow_public_source_uris=field_name == "summary",
        )


class TaskFinalizationResponse(_StrictContract):
    """返修预算耗尽后的终局成稿结果。

    这不是新的 Review：模型不再给出返修建议，而是精确覆盖当前
    所有结果组合，并基于已有证据返回可回答或不可回答的最终文本。
    """

    findings: tuple[TaskFinalizationFinding, ...]
    answerability: Literal["answerable", "unanswerable"]
    final_answer: StrictStr

    @field_validator("findings", mode="before")
    @classmethod
    def coerce_finalization_findings(cls, value: Any) -> tuple[Any, ...]:
        return _coerce_json_array_to_tuple(value, "findings")

    @field_validator("findings")
    @classmethod
    def validate_finalization_findings(
        cls,
        value: tuple[TaskFinalizationFinding, ...],
    ) -> tuple[TaskFinalizationFinding, ...]:
        if not 1 <= len(value) <= 5:
            raise ValueError("终局成稿必须覆盖一至五个当前结果")
        identities = tuple(item.hypothesis_execution_input_record_id for item in value)
        if len(set(identities)) != len(identities):
            raise ValueError("终局成稿的 hypothesis_execution_input_record_id 必须唯一")
        return value

    @field_validator("final_answer")
    @classmethod
    def validate_final_answer(cls, value: str) -> str:
        return _require_non_empty_text(
            value,
            "final_answer",
            allow_public_source_uris=True,
        )


_REFERENCE_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "uniqueItems": True,
}
_TEXT_ARRAY_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "uniqueItems": True,
}
_OBSERVATION_ARRAY_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "minItems": 1,
    "uniqueItems": True,
}


_INVESTIGATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": _JSON_SCHEMA_URI,
    "$id": "arche.investigation.response.v1",
    "title": "InvestigationResponse",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "self_judgment": {
            "type": "string",
            "enum": ["supported", "refuted"],
        },
        "narrative": {"type": "string", "minLength": 1},
        "job_refs": copy.deepcopy(_REFERENCE_SCHEMA),
        "operation_refs": copy.deepcopy(_REFERENCE_SCHEMA),
        "artifact_refs": copy.deepcopy(_REFERENCE_SCHEMA),
        "observations": copy.deepcopy(_OBSERVATION_ARRAY_SCHEMA),
        "limitations": copy.deepcopy(_TEXT_ARRAY_SCHEMA),
    },
    "required": [
        "self_judgment",
        "narrative",
        "job_refs",
        "operation_refs",
        "artifact_refs",
        "observations",
        "limitations",
    ],
}

_PASS_FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypothesis_execution_input_record_id": {"type": "string", "minLength": 1},
        "validity": {"const": "pass"},
        "hypothesis_judgment": {
            "type": "string",
            "enum": ["supported", "refuted"],
        },
    },
    "required": ["hypothesis_execution_input_record_id", "validity", "hypothesis_judgment"],
}
_FAIL_FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypothesis_execution_input_record_id": {"type": "string", "minLength": 1},
        "validity": {"const": "fail"},
        "repair_advice": {"type": "string", "minLength": 1},
    },
    "required": ["hypothesis_execution_input_record_id", "validity", "repair_advice"],
}
_SUPPORTED_FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "validity": {"const": "pass"},
        "hypothesis_judgment": {"const": "supported"},
    },
    "required": ["validity", "hypothesis_judgment"],
}

_TASK_REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": _JSON_SCHEMA_URI,
    "$id": "arche.investigation.task-review.response.v1",
    "title": "TaskReviewResponse",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "oneOf": [
                    copy.deepcopy(_PASS_FINDING_SCHEMA),
                    copy.deepcopy(_FAIL_FINDING_SCHEMA),
                ]
            },
        },
        "can_answer_original_question": {"type": "boolean"},
        "final_answer": {"type": "string", "minLength": 1},
        "answerability_explanation": {"type": "string", "minLength": 1},
    },
    "required": ["findings", "can_answer_original_question"],
    "allOf": [
        {
            "if": {
                "properties": {"can_answer_original_question": {"const": True}},
                "required": ["can_answer_original_question"],
            },
            "then": {
                "required": ["final_answer"],
                "not": {"required": ["answerability_explanation"]},
                "properties": {
                    "findings": {
                        "allOf": [
                            {
                                "not": {
                                    "contains": {
                                        "properties": {"validity": {"const": "fail"}},
                                        "required": ["validity"],
                                    }
                                }
                            },
                            {
                                "contains": copy.deepcopy(_SUPPORTED_FINDING_SCHEMA),
                                "minContains": 1,
                            },
                        ]
                    }
                },
            },
        },
        {
            "if": {
                "properties": {"can_answer_original_question": {"const": False}},
                "required": ["can_answer_original_question"],
            },
            "then": {
                "oneOf": [
                    {
                        "required": ["answerability_explanation"],
                        "not": {"required": ["final_answer"]},
                        "properties": {
                            "findings": {
                                "allOf": [
                                    {
                                        "not": {
                                            "contains": {
                                                "properties": {
                                                    "validity": {"const": "fail"}
                                                },
                                                "required": ["validity"],
                                            }
                                        }
                                    },
                                    {
                                        "contains": copy.deepcopy(_SUPPORTED_FINDING_SCHEMA),
                                        "minContains": 1,
                                    },
                                ]
                            }
                        },
                    },
                    {
                        "not": {
                            "anyOf": [
                                {"required": ["final_answer"]},
                                {"required": ["answerability_explanation"]},
                            ]
                        },
                        "properties": {
                            "findings": {
                                "anyOf": [
                                    {
                                        "contains": {
                                            "properties": {
                                                "validity": {"const": "fail"}
                                            },
                                            "required": ["validity"],
                                        },
                                        "minContains": 1,
                                    },
                                    {
                                        "not": {
                                            "contains": copy.deepcopy(
                                                _SUPPORTED_FINDING_SCHEMA
                                            )
                                        }
                                    },
                                ]
                            }
                        },
                    },
                ]
            },
        },
    ],
}

_TASK_FINALIZATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": _JSON_SCHEMA_URI,
    "$id": "arche.investigation.task-finalization.response.v1",
    "title": "TaskFinalizationResponse",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "hypothesis_execution_input_record_id": {"type": "string", "minLength": 1},
                    "summary": {"type": "string", "minLength": 1},
                },
                "required": ["hypothesis_execution_input_record_id", "summary"],
            },
        },
        "answerability": {
            "type": "string",
            "enum": ["answerable", "unanswerable"],
        },
        "final_answer": {"type": "string", "minLength": 1},
    },
    "required": ["findings", "answerability", "final_answer"],
}


# Codex Structured Outputs 的 wire 合同只保留 provider 能稳定接受的结构关键字。
# 这些 schema 不承担领域约束；响应返回后仍由 canonical schema 和 Pydantic 模型
# 执行重复项、非空文本以及其它科学合同校验。
_INVESTIGATION_PROVIDER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "self_judgment": {"type": "string", "enum": ["supported", "refuted"]},
        "narrative": {"type": "string"},
        "job_refs": {"type": "array", "items": {"type": "string"}},
        "operation_refs": {"type": "array", "items": {"type": "string"}},
        "artifact_refs": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "self_judgment",
        "narrative",
        "job_refs",
        "operation_refs",
        "artifact_refs",
        "observations",
        "limitations",
    ],
}

_TASK_REVIEW_PROVIDER_PASS_FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypothesis_execution_input_record_id": {"type": "string"},
        "validity": {"type": "string", "enum": ["pass"]},
        "hypothesis_judgment": {
            "type": "string",
            "enum": ["supported", "refuted"],
        },
    },
    "required": ["hypothesis_execution_input_record_id", "validity", "hypothesis_judgment"],
}

_TASK_REVIEW_PROVIDER_FAIL_FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypothesis_execution_input_record_id": {"type": "string"},
        "validity": {"type": "string", "enum": ["fail"]},
        "repair_advice": {"type": "string"},
    },
    "required": ["hypothesis_execution_input_record_id", "validity", "repair_advice"],
}

_TASK_REVIEW_PROVIDER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "anyOf": [
                    copy.deepcopy(_TASK_REVIEW_PROVIDER_PASS_FINDING_SCHEMA),
                    copy.deepcopy(_TASK_REVIEW_PROVIDER_FAIL_FINDING_SCHEMA),
                ]
            },
        },
        "can_answer_original_question": {"type": "boolean"},
        "final_answer": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "answerability_explanation": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
    },
    "required": [
        "findings",
        "can_answer_original_question",
        "final_answer",
        "answerability_explanation",
    ],
}

_TASK_FINALIZATION_PROVIDER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "hypothesis_execution_input_record_id": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["hypothesis_execution_input_record_id", "summary"],
            },
        },
        "answerability": {
            "type": "string",
            "enum": ["answerable", "unanswerable"],
        },
        "final_answer": {"type": "string"},
    },
    "required": ["findings", "answerability", "final_answer"],
}


def build_investigation_provider_json_schema() -> dict[str, Any]:
    """构造 Investigation provider wire 使用的保守 strict JSON Schema。

    本函数无输入、无外部副作用，只复制返回 Codex provider 边界所需的 object、array、
    string、enum、array 和 required 结构；它刻意不编码 canonical 合同中的 uniqueItems、
    非空长度等领域约束。返回值是独立副本，响应回来后仍必须使用
    ``investigation_response_json_schema`` 和领域 validator 做本地域校验。
    """

    return _copy_schema(_INVESTIGATION_PROVIDER_RESPONSE_SCHEMA)


def build_task_review_provider_json_schema() -> dict[str, Any]:
    """构造任务 Review provider wire 使用的保守 strict JSON Schema。

    本函数无输入、无外部副作用，只复制 provider 可稳定接收的 Review 结构；每个
    finding 通过嵌套 ``anyOf`` 表示 pass/fail 对象，单值状态使用 enum，两个回答文本
    字段始终 required 并以 string-or-null 表示“不适用”。返回值是独立副本；结果数量、
    组合唯一性和回答能力条件仍由 ``review_response_json_schema`` 与
    ``TaskReviewResponse`` 在本地域执行。
    """

    return _copy_schema(_TASK_REVIEW_PROVIDER_RESPONSE_SCHEMA)


def build_task_finalization_provider_json_schema() -> dict[str, Any]:
    """构造返修预算耗尽后的终局成稿 provider wire schema。"""

    return _copy_schema(_TASK_FINALIZATION_PROVIDER_RESPONSE_SCHEMA)


def investigation_response_json_schema() -> dict[str, Any]:
    """返回 Investigation turn 的独立严格 JSON Schema 副本。"""

    return _copy_schema(_INVESTIGATION_RESPONSE_SCHEMA)


def review_response_json_schema() -> dict[str, Any]:
    """返回任务级一至五项 Review 的独立严格 JSON Schema 副本。"""

    return _copy_schema(_TASK_REVIEW_RESPONSE_SCHEMA)


def finalization_response_json_schema() -> dict[str, Any]:
    """返回终局成稿结果的独立严格 JSON Schema 副本。"""

    return _copy_schema(_TASK_FINALIZATION_RESPONSE_SCHEMA)


def validate_investigation_response(value: Mapping[str, Any] | InvestigationResponse) -> InvestigationResponse:
    """严格解析调查结果；非法字段、判断或引用会抛出 Pydantic ValidationError。"""

    if isinstance(value, InvestigationResponse):
        value = value.model_dump()
    return InvestigationResponse.model_validate(value)


def validate_task_review_response(value: Mapping[str, Any] | TaskReviewResponse) -> TaskReviewResponse:
    """严格解析 Review 结果并执行全量结果屏障和回答能力条件校验。

    provider wire schema 要求 ``final_answer`` 和 ``answerability_explanation`` 始终存在，
    不适用时使用 null；进入领域模型前只移除这两个字段的 null 值。任何其它字段、非
    null 文本或非法分支都原样交给 Pydantic，因此不会因 wire 兼容层放宽约束而放行。
    """

    if isinstance(value, TaskReviewResponse):
        normalized: Any = value.model_dump()
    elif isinstance(value, Mapping):
        normalized = dict(value)
    else:
        normalized = value
    if isinstance(normalized, dict):
        for field_name in ("final_answer", "answerability_explanation"):
            if normalized.get(field_name) is None:
                normalized.pop(field_name, None)
    return TaskReviewResponse.model_validate(normalized)


def validate_task_finalization_response(
    value: Mapping[str, Any] | TaskFinalizationResponse,
) -> TaskFinalizationResponse:
    """严格解析终局成稿结果，拒绝 Review 的返修字段或额外状态。"""

    if isinstance(value, TaskFinalizationResponse):
        value = value.model_dump()
    return TaskFinalizationResponse.model_validate(value)
