"""Investigation v8 的内部事实写入边界。

本模块不对 Controller 暴露写接口。WorkingMemoryService 通过这里把来源引用校验、
不可变事件追加和当前索引更新放进同一个 SQLite 事务；上层只可把
VerifiedInvestigationEvidence 这种已由权威适配器生成的窄摘要交给服务，不能把
MiniChem receipt、私有 transcript、token 或本机路径写进 Working Memory。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

from .contracts import (
    ContractValidationError,
    canonical_json,
    validate_investigation_evidence_payload,
    validate_source_ref_shape,
)

if TYPE_CHECKING:  # pragma: no cover - 只供静态类型检查，运行时避免循环导入
    from .memory_service import WorkingMemoryService


@dataclass(frozen=True)
class VerifiedInvestigationEvidence:
    """保存权威适配器核对后的安全证据摘要，不保存原始收据或私有正文。

    输入只能来自 ``from_payload``/``from_authoritative_summary`` 的完整 mapping；
    构造动作会校验 task、batch、组合、thread、scope、turn、终态 job/operation 和
    artifact 摘要；producer-job-v2 还会逐项核对 artifact、producer job 与提交
    operation 的唯一来源链。对象只保存规范 JSON 字符串；``to_payload`` 返回新的
    字典，调用者修改它不会改变对象。输入不是合法摘要时抛出
    ContractValidationError。
    """

    _payload_json: str

    def __post_init__(self) -> None:
        if not isinstance(self._payload_json, str):
            raise ContractValidationError("VerifiedInvestigationEvidence 必须来自安全摘要")
        try:
            payload = json.loads(self._payload_json)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("VerifiedInvestigationEvidence 摘要已损坏") from exc
        normalized = validate_investigation_evidence_payload(payload)
        if canonical_json(normalized) != self._payload_json:
            raise ContractValidationError("VerifiedInvestigationEvidence 摘要未规范化")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "VerifiedInvestigationEvidence":
        """从上层验证适配器的完整摘要创建窄合同对象。

        输入是 JSON-like mapping；动作只做本地合同核验和规范化，不调用 MiniChem、
        读取密钥或访问路径；返回不可变证据对象，普通字符串或缺少终态摘要时失败。
        """

        normalized = validate_investigation_evidence_payload(payload)
        return cls(canonical_json(normalized))

    @classmethod
    def from_authoritative_summary(
        cls,
        payload: Mapping[str, Any],
    ) -> "VerifiedInvestigationEvidence":
        """从上层权威验证适配器的摘要创建安全引用对象。

        输入是适配器已经核对 task、batch、组合、thread、scope、turn 及所有终态
        job/operation/artifact 的 mapping；动作复用同一窄合同并规范 JSON，不调用
        MiniChem、不读取密钥或路径；返回不可变对象。摘要缺字段、含 transcript、
        绝对路径、未终态作业或普通字符串时抛出合同错误。
        """

        return cls.from_payload(payload)

    def to_payload(self) -> dict[str, Any]:
        """返回可用于事件写入的独立摘要副本，不暴露私有 transcript 或路径。"""

        return json.loads(self._payload_json)

    @property
    def task_id(self) -> str:
        """返回证据绑定的 task 身份。"""

        return self.to_payload()["task_id"]

    @property
    def batch_id(self) -> str:
        """返回证据绑定的 batch 身份。"""

        return self.to_payload()["batch_id"]

    @property
    def combination_id(self) -> str:
        """返回证据绑定的组合身份。"""

        return self.to_payload()["combination_id"]


class _InvestigationFactWriter:
    """由 WorkingMemoryService 使用的事件追加器，保持唯一事务写入边界。"""

    def __init__(self, service: "WorkingMemoryService") -> None:
        self._service = service

    def append_event(
        self,
        conn: Any,
        *,
        task_id: str,
        event_key: str,
        event_type: str,
        round_id: int,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        update_current_index: Callable[[str], None],
    ) -> str:
        """在一个事务中追加事实并更新当前索引。

        输入是已校验 payload、事件稳定键、任务来源引用和索引更新回调；动作先做
        同内容幂等/CAS、来源存在性和 operation 占用检查，再追加不可变事件并执行
        callback。任何失败都会抛错并由外层事务回滚，返回事件 ID；不会写入模型正文
        之外的 token、路径或 transcript。
        """

        service = self._service
        service._require_investigation_task(conn, task_id)
        normalized_refs = validate_source_ref_shape(sorted(list(source_refs)))
        normalized_refs = service._validate_source_refs(conn, task_id, normalized_refs)
        existing_id = service._existing_event_exact(
            conn,
            event_key=event_key,
            task_id=task_id,
            round_id=round_id,
            event_type=event_type,
            payload=payload,
            source_refs=normalized_refs,
        )
        if existing_id is not None:
            return existing_id
        operation_instance_id = payload.get("operation_instance_id")
        service._assert_investigation_operation_available(
            conn,
            task_id,
            operation_instance_id if isinstance(operation_instance_id, str) else None,
        )
        event_id, created = service._insert_event(
            conn,
            event_key=event_key,
            task_id=task_id,
            round_id=round_id,
            event_type=event_type,
            payload=payload,
            source_refs=normalized_refs,
            check_operation_occupancy=False,
        )
        if created:
            update_current_index(event_id)
        return event_id
