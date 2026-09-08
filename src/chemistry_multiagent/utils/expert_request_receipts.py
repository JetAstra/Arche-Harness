"""保存和恢复专家模型请求的稳定收据。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from chemistry_multiagent.utils.task_artifacts import (
    write_content_addressed_json,
    write_task_relative_json,
)


EXPERT_REQUEST_RECEIPT_SCHEMA_VERSION = "expert-request-receipt.v1"

ExpertRequestDispatchObserver = Callable[
    [str, str, Sequence[Mapping[str, Any]]], None
]


class ExpertRequestReceiptError(RuntimeError):
    """表示 provider 调用前无法持久化实际请求收据。

    调用方在 observer 写入 started 收据失败或收据字段非法时抛出此异常；异常本身
    不携带消息正文或密钥，调用方应停止发送当前请求并保存失败事实。
    """


def hash_expert_request_descriptor(request_descriptor: Mapping[str, Any]) -> str:
    """将不含 prompt 正文的专家请求描述编码为稳定 SHA-256 摘要。

    输入是可 JSON 编码的请求描述映射；动作按稳定键序列化，不写入文件也不修改
    输入。返回值是十六进制 SHA-256 字符串；字段不能严格 JSON 编码或含非有限数时
    抛出编码异常，避免生成不可复验的指纹。
    """

    encoded = json.dumps(
        dict(request_descriptor),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_expert_messages(messages: Any) -> str:
    """计算实际发送给 provider 的完整 system/user 消息数组摘要。

    输入是消息对象列表或元组；动作复制每项映射并按稳定键序列化，不保存消息正文
    或修改输入。返回值是十六进制 SHA-256 字符串；输入不是数组、成员不是对象或
    内容不能严格 JSON 编码时抛出 ``TypeError`` 或编码异常。
    """

    if not isinstance(messages, (list, tuple)):
        raise TypeError("expert messages 必须是数组")
    normalized = [
        dict(item)
        for item in messages
        if isinstance(item, Mapping)
    ]
    if len(normalized) != len(messages):
        raise TypeError("expert messages 每项必须是 object")
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_expert_backend_fingerprint(expert_agent: Any) -> str | None:
    """从冻结运行选择读取公开后端指纹，无法读取时返回 null。"""

    return get_expert_backend_fingerprints(expert_agent).get("expert")


def get_expert_backend_fingerprints(
    expert_agent: Any,
) -> dict[str, str | None]:
    """读取冻结的 expert/general 角色后端指纹。

    输入是已配置的专家代理；动作只读取公开运行选择或角色指纹，不访问消息正文和
    密钥。返回值固定包含 ``expert``、``general`` 两个实际调用角色的指纹或
    ``None``；配置缺失、
    选择无法转为映射时返回空指纹，读取过程不会把内部异常传播给调用方。
    """

    config = getattr(expert_agent, "config", None)
    role_fingerprints = getattr(config, "role_backend_fingerprints", None)
    if isinstance(role_fingerprints, Mapping):
        return {
            "expert": role_fingerprints.get("expert"),
            "general": role_fingerprints.get("general"),
        }
    selection = getattr(config, "runtime_selection", None)
    if selection is None:
        return {"expert": None, "general": None}
    if isinstance(selection, Mapping):
        public_selection = deepcopy(dict(selection))
    else:
        to_dict = getattr(selection, "to_dict", None)
        if not callable(to_dict):
            return {"expert": None, "general": None}
        try:
            public_selection = deepcopy(dict(to_dict()))
        except (TypeError, ValueError):
            return {"expert": None, "general": None}
    provider = public_selection.get("provider")
    fingerprint = (
        provider.get("config_fingerprint")
        if isinstance(provider, Mapping)
        else None
    )
    if isinstance(fingerprint, str) and len(fingerprint) == 64:
        selected_backend = public_selection.get("backend")
        if selected_backend == "fallback":
            # 没有显式角色映射时只能证明实际选中的专家后端；不能把 Flash
            # 指纹伪造为 General opinion/Judge 的 Pro 指纹。
            return {"expert": fingerprint, "general": None}
        if selected_backend == "arche_chem_7b":
            return {"expert": fingerprint, "general": None}
        return {"expert": None, "general": fingerprint}
    derived_fingerprint = hash_expert_request_descriptor(public_selection)
    selected_backend = public_selection.get("backend")
    if selected_backend == "fallback":
        return {"expert": derived_fingerprint, "general": None}
    if selected_backend == "arche_chem_7b":
        return {"expert": derived_fingerprint, "general": None}
    return {"expert": None, "general": derived_fingerprint}


def build_expert_request_receipt_bundle(
    context: Any,
    *,
    combination: Mapping[str, Any],
    operation_instance_id: str,
    phase: str,
    request_descriptor: Mapping[str, Any],
    calls: Sequence[tuple[str, str]] | None = None,
    backend_fingerprint: str | None,
) -> tuple[str, dict[str, Any]]:
    """只写入不代表 provider 已调用的编译收据。

    ``calls`` 仅为旧调用方保留兼容参数，不会被当作实际调用。实际 provider 调用
    必须由 ``build_expert_request_dispatch_observer`` 在消息已经构造后逐个追加，
    因此编译阶段不会伪造 started，也不会为未发生的 Gaussian unit 建立调用记录。
    输入是任务上下文、组合身份、阶段、请求描述和可选后端指纹；动作写入不可变的
    compiled 收据并返回内存收据，不调用 provider。返回值是收据相对路径和可继续
    追加的字典；路径或 JSON 写入失败时抛出存储异常，内存收据不会伪造成功调用。
    """

    del calls
    request_hash = hash_expert_request_descriptor(request_descriptor)
    directory = PurePosixPath(
        "working_memory",
        "expert_requests",
        operation_instance_id,
    )
    compiled_ref = write_content_addressed_json(
        context.task_root,
        directory,
        "compiled_request",
        {
            "schema_version": "expert-request.v1",
            "status": "compiled",
            "phase": phase,
            "combination_id": combination["combination_id"],
            "workflow_version": combination["workflow_version"],
            "execution_id": combination["execution_id"],
            "operation_instance_id": operation_instance_id,
            "backend_fingerprint": backend_fingerprint,
            "request_hash": request_hash,
            "calls": [],
        },
    )
    request_receipt = {
        "schema_version": EXPERT_REQUEST_RECEIPT_SCHEMA_VERSION,
        "status": "compiled",
        "phase": phase,
        "combination_id": combination["combination_id"],
        "workflow_version": combination["workflow_version"],
        "execution_id": combination["execution_id"],
        "operation_instance_id": operation_instance_id,
        "backend_fingerprint": backend_fingerprint,
        "request_hash": request_hash,
        "request_ref": compiled_ref,
        "started_ref": None,
        "calls": [],
    }
    return compiled_ref, request_receipt


def build_expert_request_dispatch_observer(
    context: Any,
    *,
    request_receipt: dict[str, Any],
    backend_fingerprint_by_role: Mapping[str, str | None] | None = None,
) -> ExpertRequestDispatchObserver:
    """构造 provider 调用前的持久化 observer。

    observer 接收已经完成的 system/user 消息，在 provider 调用前原子写入一份
    累积 started 快照。写入失败直接抛错，调用方因此不会把未留痕的请求发给
    provider；成功后才更新内存收据，后续调用会继续追加实际 call。输入是任务上下文、
    可变收据字典和按角色保存的后端指纹；返回值是 observer 回调。收据类型或字段
    非法时抛出 ``TypeError``/``ValueError``，写入失败时抛出 ``ExpertRequestReceiptError``。
    """

    if not isinstance(request_receipt, dict):
        raise TypeError("request_receipt 必须是可变 object")
    role_fingerprints = (
        dict(backend_fingerprint_by_role)
        if isinstance(backend_fingerprint_by_role, Mapping)
        else {}
    )

    def observe(
        operation: str,
        role: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> None:
        """在 provider 调用前持久化一笔实际请求。

        输入是操作名、角色和已构造的 system/user 消息；动作计算消息摘要，追加
        ``calls`` 并原子写入 started 快照，再更新内存收据，不触发 provider。返回值
        为 ``None``；收据已进入终局、参数非法、消息不能编码或文件写入失败时抛出
        ``ExpertRequestReceiptError``，调用方必须停止发送该请求。
        """

        try:
            if request_receipt.get("status") not in {"compiled", "started"}:
                raise ValueError("专家请求收据已经进入终局，不能追加 provider 调用")
            if not isinstance(operation, str) or not operation:
                raise ValueError("专家请求 operation 必须是非空字符串")
            if role not in {"expert", "general"}:
                raise ValueError("专家请求 role 非法")
            message_hash = hash_expert_messages(messages)
            calls = request_receipt.get("calls")
            if not isinstance(calls, list):
                raise ValueError("专家请求收据 calls 必须是数组")
            call_index = len(calls) + 1
            call = {
                "call_id": f"{request_receipt['phase']}-{call_index}",
                "call_index": call_index,
                "operation": operation,
                "role": role,
                "message_hash": message_hash,
                "request_hash": hash_expert_request_descriptor(
                    {
                        "request_hash": request_receipt["request_hash"],
                        "operation": operation,
                        "role": role,
                        "call_index": call_index,
                        "message_hash": message_hash,
                    }
                ),
                "backend_fingerprint": role_fingerprints.get(
                    role,
                    request_receipt.get("backend_fingerprint"),
                ),
            }
            next_calls = [deepcopy(item) for item in calls]
            next_calls.append(call)
            started_payload = deepcopy(dict(request_receipt))
            started_payload["status"] = "started"
            started_payload["calls"] = next_calls
            directory = PurePosixPath(
                "working_memory",
                "expert_requests",
                str(request_receipt["operation_instance_id"]),
            )
            started_ref = (
                directory / f"started_request_{call_index:04d}.json"
            ).as_posix()
            started_payload["started_ref"] = started_ref
            write_task_relative_json(
                context.task_root,
                started_ref,
                started_payload,
            )
        except ExpertRequestReceiptError:
            raise
        except Exception as exc:
            raise ExpertRequestReceiptError("无法持久化专家实际请求 started 收据") from exc
        request_receipt["status"] = "started"
        request_receipt["started_ref"] = started_ref
        request_receipt["calls"] = next_calls

    return observe


def get_expert_request_started_ref(
    request_receipt: Mapping[str, Any] | None,
) -> str | None:
    """读取最近一次实际 started 快照路径。

    输入是可选专家请求收据；动作只读取 ``started_ref``，不修改收据或文件。返回值
    是相对路径字符串；收据缺失、字段不是字符串或仅完成编译尚未 dispatch 时返回
    ``None``，不会抛出原始解析异常。
    """

    if not isinstance(request_receipt, Mapping):
        return None
    started_ref = request_receipt.get("started_ref")
    return started_ref if isinstance(started_ref, str) else None


def write_expert_request_terminal_receipt(
    context: Any,
    *,
    started_receipt_ref: str | None,
    request_receipt: Mapping[str, Any] | None,
    status: str,
) -> str | None:
    """在 provider 返回或明确失败后写入持久终局收据，关闭 crash 扫描窗口。

    调用方传入可变字典时同步更新其状态，使随后写入 Review、Debate 或暂停证据
    的内嵌收据与磁盘终局文件保持一致。
    """

    if not started_receipt_ref or not isinstance(request_receipt, Mapping):
        return None
    if status not in {"completed", "failed"}:
        raise ValueError("专家请求终局收据 status 非法")
    terminal = deepcopy(dict(request_receipt))
    terminal["status"] = status
    terminal_ref = write_content_addressed_json(
        context.task_root,
        PurePosixPath(started_receipt_ref).parent,
        "terminal_request",
        terminal,
    )
    if isinstance(request_receipt, dict):
        request_receipt["status"] = status
    return terminal_ref


def find_started_expert_request_receipt(
    context: Any,
    *,
    combination: Mapping[str, Any],
    phase: str,
    request_hash: str,
) -> tuple[str, dict[str, Any]] | None:
    """扫描当前任务目录中同一请求的 started 收据，阻止未知在途请求重发。"""

    root = Path(context.task_root) / "working_memory" / "expert_requests"
    if not root.is_dir() or root.is_symlink():
        return None
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for path in sorted(root.rglob("started_request_*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if (
            payload.get("schema_version") != EXPERT_REQUEST_RECEIPT_SCHEMA_VERSION
            or payload.get("status") != "started"
            or payload.get("phase") != phase
            or payload.get("combination_id") != combination.get("combination_id")
            or payload.get("workflow_version") != combination.get("workflow_version")
            or payload.get("execution_id") != combination.get("execution_id")
            or payload.get("request_hash") != request_hash
        ):
            continue
        terminal_found = False
        for terminal_path in sorted(path.parent.glob("terminal_request_*.json")):
            if terminal_path.is_symlink() or not terminal_path.is_file():
                continue
            try:
                terminal_payload = json.loads(terminal_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                continue
            if (
                isinstance(terminal_payload, Mapping)
                and terminal_payload.get("status") in {"completed", "failed"}
                and terminal_payload.get("operation_instance_id")
                == payload.get("operation_instance_id")
                and terminal_payload.get("request_hash") == request_hash
            ):
                terminal_found = True
                break
        if terminal_found:
            continue
        calls = payload.get("calls")
        if not isinstance(calls, list) or not calls:
            continue
        try:
            reference = path.resolve().relative_to(Path(context.task_root).resolve()).as_posix()
        except ValueError:
            continue
        candidates.append((len(calls), reference, deepcopy(dict(payload))))
    if candidates:
        _, reference, payload = max(candidates, key=lambda item: (item[0], item[1]))
        return reference, payload
    return None


__all__ = [
    "EXPERT_REQUEST_RECEIPT_SCHEMA_VERSION",
    "ExpertRequestReceiptError",
    "ExpertRequestDispatchObserver",
    "build_expert_request_dispatch_observer",
    "build_expert_request_receipt_bundle",
    "find_started_expert_request_receipt",
    "get_expert_backend_fingerprints",
    "get_expert_request_started_ref",
    "get_expert_backend_fingerprint",
    "hash_expert_messages",
    "hash_expert_request_descriptor",
    "write_expert_request_terminal_receipt",
]
