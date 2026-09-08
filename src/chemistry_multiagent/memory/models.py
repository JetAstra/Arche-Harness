"""集中放置新旧记忆记录都会用到的常量和轻量转换函数。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


# v10 在 v9 的 Working Memory 上增量加入 Investigation turn 等待回执和
# resumable attempt 状态；v9 仍保存不可变终止事实、replacement attempt 游标和
# 任务级 Review 活动指针；
# v8 在 v7 的 Working Memory 上增量加入任务级 Review 会话、返修 turn 收据、
# 权威证据摘要和稳定的 Investigation 当前状态。旧 v5/v6/v7 事件仍由迁移逻辑读取，
# 不删除或伪造历史事件。
SCHEMA_VERSION = 10
VERIFICATION_PENDING = "pending"

# provider turn 的一次尝试只能处于以下五种状态；终止状态只能由
# investigation_turn_attempt_terminated 事件产生，不能从缺失 Result 推断。
TURN_ATTEMPT_STATUSES = frozenset(
    {
        "in_flight",
        "resumable",
        "succeeded",
        "terminal_failed",
        "terminal_interrupted",
    }
)
# Investigation initial/repair 才允许把 attempt 标为 resumable；Review 仍只有
# in_flight 和四类结果/终止状态，数据库 CHECK 与 payload 合同都按这两个集合执行。
INVESTIGATION_TURN_ATTEMPT_STATUSES = TURN_ATTEMPT_STATUSES
REVIEW_TURN_ATTEMPT_STATUSES = frozenset(
    {"in_flight", "succeeded", "terminal_failed", "terminal_interrupted"}
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
