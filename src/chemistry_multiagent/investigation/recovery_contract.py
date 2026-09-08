"""Investigation V2 恢复核验的有限结果合同。

恢复时不能把“当前进程没有把某个投影写回来”和“两个权威身份确实冲突”
混为一谈。本模块把这两个判断以及外部暂时不可用收敛为五个显式结果。
调用方只需要检查 ``status`` 和 ``can_continue``/``must_suspend``，不需要猜测
``None``、``unknown`` 或自由文本的含义。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Self


class RecoveryContractError(ValueError):
    """恢复核验结果或输入身份不满足合同。"""


class RecoveryStatus(str, Enum):
    """恢复核验允许的五个互斥结果。"""

    CONSISTENT = "Consistent"
    RECONCILED = "Reconciled"
    TEMPORARILY_UNAVAILABLE = "TemporarilyUnavailable"
    RECOVERY_EVIDENCE_MISSING = "RecoveryEvidenceMissing"
    VERIFIED_INTEGRITY_CONFLICT = "VerifiedIntegrityConflict"


# 名称别名保留规范中的首字母大写写法，便于调用方直接构造结果。
RecoveryVerificationStatus = RecoveryStatus

_SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _require_code(value: Any, label: str) -> str:
    """校验结构化原因码，避免把异常正文写入恢复合同。"""

    if not isinstance(value, str) or not value.strip():
        raise RecoveryContractError(f"{label} 必须是非空字符串")
    value = value.strip()
    if _SAFE_CODE.fullmatch(value) is None:
        raise RecoveryContractError(f"{label} 必须是安全结构化代码")
    return value


def _require_fields(value: Any, label: str) -> tuple[str, ...]:
    """校验发生协调的字段名称是有序且唯一的字符串序列。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecoveryContractError(f"{label} 必须是字符串序列")
    fields = tuple(_require_code(item, f"{label}[]") for item in value)
    if len(fields) != len(set(fields)):
        raise RecoveryContractError(f"{label} 不能包含重复字段")
    return fields


@dataclass(frozen=True, slots=True)
class RecoveryVerification:
    """一次恢复核验的公开结果。

    ``reconciled_fields`` 只描述已明确允许从权威来源协调的字段；它不能把
    缺失或未注册字段自动升级为一致，也不能把身份冲突降级为协调成功。
    """

    status: RecoveryStatus
    reason_code: str
    reconciled_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """规范化状态和结构化原因。"""

        status = self.status
        if not isinstance(status, RecoveryStatus):
            try:
                status = RecoveryStatus(status)
            except (TypeError, ValueError) as exc:
                raise RecoveryContractError("恢复核验 status 未注册") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", _require_code(self.reason_code, "reason_code"))
        object.__setattr__(
            self,
            "reconciled_fields",
            _require_fields(self.reconciled_fields, "reconciled_fields"),
        )
        if status is not RecoveryStatus.RECONCILED and self.reconciled_fields:
            raise RecoveryContractError(
                "只有 Reconciled 结果可以携带 reconciled_fields"
            )

    @property
    def can_continue(self) -> bool:
        """结果是否允许公开恢复 seam 继续或重放。"""

        return self.status in {
            RecoveryStatus.CONSISTENT,
            RecoveryStatus.RECONCILED,
        }

    @property
    def must_suspend(self) -> bool:
        """结果是否必须保留 Open 状态并等待后续恢复。"""

        return self.status in {
            RecoveryStatus.TEMPORARILY_UNAVAILABLE,
            RecoveryStatus.RECOVERY_EVIDENCE_MISSING,
        }

    @property
    def must_fail_integrity(self) -> bool:
        """结果是否足以写入明确的完整性失败事实。"""

        return self.status is RecoveryStatus.VERIFIED_INTEGRITY_CONFLICT

    def to_dict(self) -> dict[str, Any]:
        """返回可审计但不含异常正文的稳定结构。"""

        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "reconciled_fields": list(self.reconciled_fields),
        }

    as_mapping = to_dict

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        """严格读取持久化核验结果；不接受 unknown 或额外字段。"""

        if not isinstance(value, Mapping):
            raise RecoveryContractError("恢复核验必须是 JSON object")
        expected = {"status", "reason_code", "reconciled_fields"}
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details: list[str] = []
            if missing:
                details.append(f"缺少 {', '.join(missing)}")
            if extra:
                details.append(f"含额外字段 {', '.join(extra)}")
            raise RecoveryContractError(f"恢复核验字段合同不匹配：{'；'.join(details)}")
        return cls(
            status=value["status"],
            reason_code=value["reason_code"],
            reconciled_fields=tuple(value["reconciled_fields"]),
        )


@dataclass(frozen=True, slots=True)
class Consistent(RecoveryVerification):
    """权威身份完全一致，可以继续。"""

    status: RecoveryStatus = field(init=False, default=RecoveryStatus.CONSISTENT)


@dataclass(frozen=True, slots=True)
class Reconciled(RecoveryVerification):
    """有限字段已从权威来源协调，可以继续或重放。"""

    status: RecoveryStatus = field(init=False, default=RecoveryStatus.RECONCILED)


@dataclass(frozen=True, slots=True)
class TemporarilyUnavailable(RecoveryVerification):
    """外部拥有者暂时不可用，保持 Open。"""

    status: RecoveryStatus = field(
        init=False,
        default=RecoveryStatus.TEMPORARILY_UNAVAILABLE,
    )


@dataclass(frozen=True, slots=True)
class RecoveryEvidenceMissing(RecoveryVerification):
    """恢复所需证据尚未可读，保持 Open。"""

    status: RecoveryStatus = field(
        init=False,
        default=RecoveryStatus.RECOVERY_EVIDENCE_MISSING,
    )


@dataclass(frozen=True, slots=True)
class VerifiedIntegrityConflict(RecoveryVerification):
    """两个权威来源确认身份发生冲突，允许完整性失败。"""

    status: RecoveryStatus = field(
        init=False,
        default=RecoveryStatus.VERIFIED_INTEGRITY_CONFLICT,
    )


RecoveryResult = (
    Consistent
    | Reconciled
    | TemporarilyUnavailable
    | RecoveryEvidenceMissing
    | VerifiedIntegrityConflict
)


def reconcile_recovery_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    required_fields: Sequence[str],
    reconciled_fields: Sequence[str] = (),
    ignored_fields: Sequence[str] = (),
) -> RecoveryResult:
    """比较两份公开身份，并区分缺失、允许协调和真实冲突。

    ``expected`` 和 ``actual`` 都必须由调用方从各自权威 owner 读取。缺少必需
    字段返回 ``RecoveryEvidenceMissing``；只有必需字段同时存在且不相等时，才
    返回 ``VerifiedIntegrityConflict``。展示字段可以通过 ``ignored_fields`` 明确
    排除，但不能通过这个参数绕过必需身份字段核对。
    """

    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise RecoveryContractError("恢复身份必须是 JSON object")
    required = _require_fields(required_fields, "required_fields")
    allowed_reconcile = set(_require_fields(reconciled_fields, "reconciled_fields"))
    ignored = set(_require_fields(ignored_fields, "ignored_fields"))
    missing = [
        field_name
        for field_name in required
        if field_name not in expected or field_name not in actual
    ]
    if missing:
        return RecoveryEvidenceMissing(
            reason_code="recovery_evidence_missing",
        )
    changed = [
        field_name
        for field_name in required
        if field_name not in ignored and expected[field_name] != actual[field_name]
    ]
    if changed:
        if all(field_name in allowed_reconcile for field_name in changed):
            return Reconciled(
                reason_code="recovery_identity_reconciled",
                reconciled_fields=tuple(changed),
            )
        return VerifiedIntegrityConflict(reason_code="verified_identity_conflict")
    return Consistent(reason_code="recovery_identity_match")


__all__ = [
    "Consistent",
    "RecoveryContractError",
    "RecoveryEvidenceMissing",
    "RecoveryResult",
    "RecoveryStatus",
    "RecoveryVerification",
    "RecoveryVerificationStatus",
    "Reconciled",
    "TemporarilyUnavailable",
    "VerifiedIntegrityConflict",
    "reconcile_recovery_identity",
]
