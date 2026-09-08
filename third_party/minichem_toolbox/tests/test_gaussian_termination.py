"""Gaussian 归档警告共享判定的成功门和失败反例。"""

from __future__ import annotations

import pytest

from minichem_toolbox.gaussian_termination import (
    GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER,
    GAUSSIAN_ARCHIVE_PUNCHED_MARKER,
    gaussian_archive_only_completion,
    gaussian_archive_warning_provenance,
)


def _complete_archive_warning(**updates: object) -> dict[str, object]:
    """创建满足共享 Gaussian 归档警告成功门的独立输出记录。"""

    completed: dict[str, object] = {
        "returncode": 1,
        "stdout": (
            f"{GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER}\n"
            f"{GAUSSIAN_ARCHIVE_PUNCHED_MARKER}\n"
            "Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.\n"
        ),
        "stderr": "",
    }
    completed.update(updates)
    return completed


def test_gaussian_archive_only_completion_returns_structured_provenance() -> None:
    """严格归档警告输出通过判定，并保留返回码和两个原始 marker。"""

    assert gaussian_archive_only_completion(_complete_archive_warning()) is True
    assert gaussian_archive_warning_provenance() == {
        "gaussian_external_returncode": 1,
        "gaussian_termination": "normal",
        "gaussian_archive_warning": {
            "entry_marker": GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER,
            "punched_marker": GAUSSIAN_ARCHIVE_PUNCHED_MARKER,
        },
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"returncode": 0},
        {"returncode": 2},
        {"returncode": -9},
        {"returncode": True},
        {"stdout": _complete_archive_warning()["stdout"].replace(
            f"{GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER}\n", ""
        )},
        {"stdout": _complete_archive_warning()["stdout"].replace(
            f"{GAUSSIAN_ARCHIVE_PUNCHED_MARKER}\n", ""
        )},
        {"stdout": "Normal termination of Gaussian 16\n"},
        *[
            {"stdout": _complete_archive_warning()["stdout"] + f"{marker}\n"}
            for marker in (
                "Error termination via Lnk1e",
                "Erroneous termination",
                "QPErr --- input syntax error",
                "Fatal Error: failed",
                "segmentation fault",
                "SIGTERM",
                "Killed",
                "Aborted",
            )
        ],
        {"stdout": _complete_archive_warning()["stdout"], "stderr": "QPErr\n"},
        {"timed_out": True},
        {"memory_exceeded": True},
        {"cancelled": True},
        {"cleanup_pending": True},
        {"process_group_activity": "nonempty"},
        {"background_process_cleanup": "remaining_process_group"},
    ],
)
def test_gaussian_archive_only_completion_rejects_unsafe_records(
    updates: dict[str, object],
) -> None:
    """任何退出码、输出或进程状态不满足完整合同都必须拒绝。"""

    assert gaussian_archive_only_completion(_complete_archive_warning(**updates)) is False
