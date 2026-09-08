"""六个 research_task CLI 动作和公共启动脚本的控制面回归。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from chemistry_multiagent.controllers import chemistry_multiagent_controller as controller_module


def _fake_result(research_task_id: str) -> dict[str, object]:
    """构造公开 CLI 结果，避免测试触碰模型或 Working Memory。"""

    return {"status": "open", "research_task_id": research_task_id}


class _FakeController:
    """记录六个公开动作调用的最小 Controller。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def create_standard_research_task(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("create-standard", (), kwargs))
        return _fake_result("standard-task")

    def create_author_hypothesis_task(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("create-author", (), kwargs))
        return _fake_result("author-task")

    def create_top5_confirmation_task(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("create-top5", (), kwargs))
        return _fake_result("top5-task")

    def get_hypothesis_confirmation(self, *args: object) -> dict[str, object]:
        self.calls.append(("get-confirmation", args, {}))
        return _fake_result(str(args[0]))

    def submit_hypothesis_confirmation_and_continue(
        self,
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append(("submit-confirmation", args, kwargs))
        return _fake_result(str(args[0]))

    def resume_research_task(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls.append(("resume", args, kwargs))
        return _fake_result(str(args[0]))


def test_cli_dispatches_only_the_six_formal_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI 按正式 action 分派，不再解析旧 run mode 或恢复选项。"""

    fake = _FakeController()
    monkeypatch.setattr(
        controller_module,
        "_controller_from_cli_args",
        lambda _args: fake,
    )
    monkeypatch.setattr(
        controller_module,
        "_load_strict_author_hypotheses_json",
        lambda _path: [{"hypothesis": "作者假设"}],
    )
    monkeypatch.setattr(
        controller_module,
        "_load_hypothesis_confirmation_request",
        lambda _path: (
            {
                "confirmation_id": "confirmation-1",
                "expected_research_task_sequence": 1,
                "decisions": [],
            },
            {},
        ),
    )

    controller_module.run_from_args(
        [
            "create-standard-research-task",
            "--question",
            "标准研究问题",
            "--no-search",
        ]
    )
    controller_module.run_from_args(
        [
            "create-author-hypothesis-task",
            "--question",
            "作者问题",
            "--author-hypotheses-json",
            str(tmp_path / "author.json"),
        ]
    )
    controller_module.run_from_args(
        [
            "create-top5-confirmation-task",
            "--question",
            "Top5 问题",
            "--no-search",
        ]
    )
    controller_module.run_from_args(["get-hypothesis-confirmation", "top5-task"])
    controller_module.run_from_args(
        [
            "submit-hypothesis-confirmation-and-continue",
            "top5-task",
            "--confirmation-request-json",
            str(tmp_path / "confirmation.json"),
        ]
    )
    controller_module.run_from_args(["resume-research-task", "top5-task"])

    assert [name for name, _args, _kwargs in fake.calls] == [
        "create-standard",
        "create-author",
        "create-top5",
        "get-confirmation",
        "submit-confirmation",
        "resume",
    ]
    assert fake.calls[0][2]["scientific_question"] == "标准研究问题"
    assert fake.calls[1][2]["author_hypotheses"] == [{"hypothesis": "作者假设"}]
    assert fake.calls[3][1] == ("top5-task",)
    assert fake.calls[4][1] == ("top5-task",)
    assert fake.calls[4][2]["confirmation_id"] == "confirmation-1"


def test_cli_rejects_removed_legacy_options() -> None:
    """旧 run mode 和旧恢复 flag 不能成为公开 CLI alias。"""

    parser = controller_module._build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-mode", "end_to_end", "--question", "旧入口"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--resume-task", "old-task"])


def test_public_scripts_use_formal_actions_and_shell_syntax() -> None:
    """两个保留文件名的公共脚本调用正式 action，并可通过 Bash 语法检查。"""

    root = Path(__file__).parents[1]
    expected = {
        "run_end_to_end.sh": "create-standard-research-task",
        "run_to_hypothesis_top5.sh": "create-top5-confirmation-task",
    }
    for name, action in expected.items():
        path = root / "scripts" / name
        source = path.read_text(encoding="utf-8")
        assert action in source
        assert "--run-mode" not in source
        assert "--resume-task" not in source
        assert "ARCHE_RUN_STORAGE_ROOT" in source
        assert "ARCHE_STAGE_HARNESS_WORKSPACE_ROOT" in source
        assert "ARCHE_STAGE_HARNESS_TRACE_ROOT" in source
        assert "outputs/runs" not in source
        subprocess.run(["bash", "-n", str(path)], check=True)
