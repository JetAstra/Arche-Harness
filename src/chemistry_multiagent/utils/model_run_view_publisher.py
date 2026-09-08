"""把模型阶段视图交给独立 Writer 原子发布。

``ModelRunViewPublisher`` 只负责把确定性视图构造函数和
``ModelRunOutputWriter`` 接起来。它不参与科学任务状态、不读取原始模型文件，也不
创建第二套恢复数据。任何构造、读取或写入异常都会通知 Writer 标记展示降级并返回
``False``，从而让正式工作流可以继续运行。
"""

from __future__ import annotations

from copy import deepcopy
import json
import posixpath
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from chemistry_multiagent.utils.model_run_outputs import ModelRunOutputWriter
from chemistry_multiagent.utils.model_run_views import (
    construct_hypothesis_outcomes,
    construct_hypothesis_summary,
    construct_revision_summary,
    construct_router_summary,
    construct_workflow_version,
)


class ModelRunViewPublisher:
    """协调模型展示视图的构造、路径选择和原子写入。"""

    def __init__(self, writer: ModelRunOutputWriter) -> None:
        """保存展示 Writer；不创建目录、不读取已有文件。"""

        if not isinstance(writer, ModelRunOutputWriter):
            raise TypeError("writer 必须是 ModelRunOutputWriter")
        self.writer = writer

    @staticmethod
    def _batch_path(batch_index: int) -> str:
        """返回三位编号的批次目录片段。"""

        return f"batch_{batch_index:03d}"

    @staticmethod
    def _combination_path(combination_index: int) -> str:
        """返回三位编号的组合目录片段。"""

        return f"combination_{combination_index:03d}"

    @staticmethod
    def _root_source_path(value: Any, target_parent: str) -> Any:
        """把模型运行根目录引用转成目标展示文件目录相对路径。"""

        if not isinstance(value, str) or not value:
            return value
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts:
            return value
        root_directories = {
            "01_retrieval",
            "02_query",
            "03_hypothesis",
            "04_planning",
            "05_router",
            "06_revision",
            "07_expert_review",
            "working_memory",
        }
        if path.parts[0] not in root_directories:
            return value
        return posixpath.relpath(path.as_posix(), target_parent)

    @classmethod
    def _normalise_source_fields(cls, value: Any, target_parent: str) -> Any:
        """递归转换 source_file/source_model_file，不改动其余展示正文。"""

        if isinstance(value, Mapping):
            return {
                str(key): (
                    cls._root_source_path(item, target_parent)
                    if key in {"source_file", "source_model_file"}
                    else cls._normalise_source_fields(item, target_parent)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._normalise_source_fields(item, target_parent) for item in value]
        if isinstance(value, tuple):
            return [cls._normalise_source_fields(item, target_parent) for item in value]
        return deepcopy(value)

    @classmethod
    def _normalise_source_files(
        cls,
        source_files: Sequence[str],
        target_parent: str,
    ) -> list[str]:
        """把来源文件列表中的根目录引用转换为目标目录相对引用。"""

        return [
            cls._root_source_path(source_file, target_parent)
            for source_file in source_files
        ]

    def _validate_source_reference(
        self,
        reference: Any,
        *,
        target_relative_path: str,
        field: str,
        allow_none: bool = False,
    ) -> None:
        """确认展示引用是相对目标文件目录且解析后仍在当前展示根内。"""

        if allow_none and reference is None:
            return
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"{field} 必须是非空相对引用")
        if "\\" in reference:
            raise ValueError(f"{field} 不得包含反斜杠")
        posix_path = PurePosixPath(reference)
        windows_path = PureWindowsPath(reference)
        if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise ValueError(f"{field} 必须是相对引用")
        root = self.writer.output_root.resolve(strict=False)
        target_parent = self.writer.resolve_output_path(target_relative_path).parent.resolve(
            strict=False
        )
        resolved = (target_parent / posix_path).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{field} 解析后越出模型展示目录") from error

    def _validate_source_file_list(
        self,
        source_files: Sequence[str],
        *,
        target_relative_path: str,
    ) -> None:
        """验证 summary/ranking 使用的来源引用列表。"""

        for index, reference in enumerate(source_files):
            self._validate_source_reference(
                reference,
                target_relative_path=target_relative_path,
                field=f"source_files[{index}]",
            )

    def _validate_source_file_map(
        self,
        source_file_map: Mapping[str, Mapping[str, Any]],
        *,
        target_relative_path: str,
    ) -> None:
        """验证每个 Hypothesis 的 generation/planning/router 来源引用。"""

        for hypothesis_id, references in source_file_map.items():
            if not isinstance(references, Mapping):
                raise ValueError(f"source_file_map[{hypothesis_id}] 必须是 object")
            for stage in ("generation", "planning", "router"):
                self._validate_source_reference(
                    references.get(stage),
                    target_relative_path=target_relative_path,
                    field=f"source_file_map[{hypothesis_id}].{stage}",
                    allow_none=True,
                )

    def _read_previous_workflow_version(
        self,
        relative_path: str,
        *,
        expected_version: int,
        expected_batch_index: int,
        expected_combination_index: int,
    ) -> Mapping[str, Any]:
        """读取同一组合的上一版 Workflow，并核对不可变身份链。"""

        root = self.writer.output_root.resolve(strict=False)
        target = self.writer.resolve_output_path(relative_path)
        if target.is_symlink() or not target.is_file():
            raise ValueError("workflow version 缺少同一组合的上一版本")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception as error:
            raise ValueError("workflow version 上一版本不是可解析 JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError("workflow version 上一版本必须是 JSON object")
        if payload.get("research_task_id") != self.writer.research_task_id:
            raise ValueError("workflow version 上一版本 research_task_id 不一致")
        if payload.get("batch_index") != expected_batch_index:
            raise ValueError("workflow version 上一版本 batch_index 不一致")
        if payload.get("combination_index") != expected_combination_index:
            raise ValueError("workflow version 上一版本 combination_index 不一致")
        if payload.get("workflow_version") != expected_version:
            raise ValueError("workflow version 上一版本编号不连续")
        return payload

    def _validate_workflow_version_chain(
        self,
        record: Mapping[str, Any],
        view: Mapping[str, Any],
        target_relative_path: str,
    ) -> None:
        """确认版本 N 只能接在同 batch/combination 的版本 N-1 后。"""

        version = view["workflow_version"]
        if version == 1:
            if view["source_type"] != "initial_planning":
                raise ValueError("version_001 只能来自 initial_planning")
            return
        if view["source_type"] != "revision_patch_applied":
            raise ValueError("version_001 之后的版本必须来自 revision_patch_applied")
        before = record.get("workflow_version_before")
        after = record.get("workflow_version_after")
        if before != version - 1 or after != version:
            raise ValueError("Revision before/after 与目标 workflow version 不一致")
        target = PurePosixPath(target_relative_path)
        previous_relative_path = (
            target.parent / f"version_{version - 1:03d}.json"
        ).as_posix()
        self._read_previous_workflow_version(
            previous_relative_path,
            expected_version=version - 1,
            expected_batch_index=view["batch_index"],
            expected_combination_index=view["combination_index"],
        )

    def _mark_degraded(self, _error: Exception | str) -> bool:
        """用固定原因通知 Writer 展示失败，不公开内部异常正文。"""

        try:
            self.writer.mark_model_output_degraded("model_view_publish_failed")
        except Exception:
            pass
        return False

    def _write_summary(self, relative_path: str, payload: Mapping[str, Any]) -> bool:
        """原子写入可更新的 summary，并把 Writer 的 False 视为展示失败。"""

        try:
            written = self.writer.update_summary(relative_path, deepcopy(dict(payload)))
        except Exception as error:
            return self._mark_degraded(error)
        if not written:
            return self._mark_degraded("模型阶段 summary 写入失败")
        return True

    def _write_outcomes(self, relative_path: str, payload: Mapping[str, Any]) -> bool:
        """原子写入 Hypothesis outcomes。"""

        try:
            written = self.writer.update_outcomes(relative_path, deepcopy(dict(payload)))
        except Exception as error:
            return self._mark_degraded(error)
        if not written:
            return self._mark_degraded("Hypothesis outcomes 写入失败")
        return True

    @staticmethod
    def _construct_round_summary(
        phase: str,
        history: Mapping[str, Any],
    ) -> dict[str, Any]:
        """构造指定模型阶段的轮次视图并保持阶段专属首轮规则。

        输入是已经合并的 Router 或 Revision 轮次记录；动作是只对 Router 展示
        明确允许非 1 首轮，再调用对应的只读视图构造器。返回精简后的阶段摘要；
        阶段未知、轮次缺失或顺序不合法时抛出视图错误，由发布入口标记展示降级。
        Revision 始终使用默认的从 1 开始规则。
        """

        if phase == "router":
            return construct_router_summary(
                history,
                allow_arbitrary_initial_round=True,
            )
        if phase == "revision":
            return construct_revision_summary(history)
        raise ValueError("未知模型阶段")

    def publish_hypothesis_summary(
        self,
        batch: Mapping[str, Any],
        source_files: Sequence[str],
        merge_decisions: Sequence[Mapping[str, Any]],
        *,
        write_ranking: bool = False,
    ) -> bool:
        """发布 Hypothesis summary，并按需创建不可变最终排名文件。

        ``summary.json`` 可以在恢复时更新；可选的 ``write_ranking=True`` 会把同一
        次排名写入 ``ranking/final_ranking.json``，该文件只允许首次创建。所有路径
        均由批次正式编号生成，不接受模型提供的目录名。
        """

        try:
            # 先保留调用方传入的运行根目录引用。summary 和 ranking 位于不同目录，
            # ranking 不能拿 summary 已经归一化后的字符串再次猜测来源位置。
            original_source_files = deepcopy(list(source_files))
            original_merge_decisions = deepcopy(list(merge_decisions))
            target_parent = (
                f"03_hypothesis/{self._batch_path(batch['batch_index'])}"
            )
            view = construct_hypothesis_summary(
                batch,
                source_files=self._normalise_source_files(
                    original_source_files,
                    target_parent,
                ),
                merge_decisions=self._normalise_source_fields(
                    original_merge_decisions,
                    target_parent,
                ),
            )
            batch_index = view["batch_index"]
            summary_path = (
                f"03_hypothesis/{self._batch_path(batch_index)}/summary.json"
            )
            self._validate_source_file_list(
                view["source_files"],
                target_relative_path=summary_path,
            )
            if not self._write_summary(summary_path, view):
                return False
            if not write_ranking:
                return True
            ranking_parent = (
                f"03_hypothesis/{self._batch_path(batch_index)}/ranking"
            )
            ranking_payload = {
                "schema_version": view["schema_version"],
                "batch_index": batch_index,
                "ranked_hypotheses": deepcopy(view["ranked_hypotheses"]),
                "source_files": self._normalise_source_files(
                    original_source_files,
                    ranking_parent,
                ),
            }
            ranking_path = (
                f"03_hypothesis/{self._batch_path(batch_index)}"
                "/ranking/final_ranking.json"
            )
            self._validate_source_file_list(
                ranking_payload["source_files"],
                target_relative_path=ranking_path,
            )
            written = self.writer.write_immutable_view(ranking_path, ranking_payload)
            if not written:
                return self._mark_degraded("Hypothesis final ranking 写入失败")
            return True
        except Exception as error:
            return self._mark_degraded(error)

    def publish_hypothesis_outcomes(
        self,
        batch: Mapping[str, Any],
        combination_facts: Sequence[Mapping[str, Any]],
        source_file_map: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        """发布当前批次的 Hypothesis outcomes。"""

        try:
            outcomes_path = (
                f"03_hypothesis/{self._batch_path(batch['batch_index'])}/outcomes.json"
                if isinstance(batch, Mapping) and isinstance(batch.get("batch_index"), int)
                else "03_hypothesis/batch_000/outcomes.json"
            )
            source_parent = PurePosixPath(outcomes_path).parent.as_posix()
            source_file_map = {
                hypothesis_id: {
                    stage: self._root_source_path(reference, source_parent)
                    for stage, reference in references.items()
                }
                for hypothesis_id, references in source_file_map.items()
            }
            self._validate_source_file_map(
                source_file_map,
                target_relative_path=outcomes_path,
            )
            view = construct_hypothesis_outcomes(
                batch,
                combination_facts,
                source_files=source_file_map,
            )
            path = (
                f"03_hypothesis/{self._batch_path(view['batch_index'])}/outcomes.json"
            )
            return self._write_outcomes(path, view)
        except Exception as error:
            return self._mark_degraded(error)

    def publish_workflow_version(self, record: Mapping[str, Any]) -> bool:
        """发布一份不可变的完整 Workflow 版本。"""

        try:
            if not isinstance(record, Mapping):
                raise TypeError("workflow version record 必须是 object")
            if record.get("research_task_id") != self.writer.research_task_id:
                raise ValueError(
                    "workflow version record.research_task_id 与 writer 不一致"
                )
            target_parent = (
                f"04_planning/{self._batch_path(record['batch_index'])}/"
                f"{self._combination_path(record['combination_index'])}/"
                "workflow_versions"
            )
            view = construct_workflow_version(
                self._normalise_source_fields(record, target_parent)
            )
            path = (
                f"04_planning/{self._batch_path(view['batch_index'])}/"
                f"{self._combination_path(view['combination_index'])}/"
                "workflow_versions/"
                f"version_{view['workflow_version']:03d}.json"
            )
            self._validate_source_reference(
                view["source_model_file"],
                target_relative_path=path,
                field="source_model_file",
            )
            self._validate_workflow_version_chain(record, view, path)
            written = self.writer.write_immutable_view(path, view)
            if not written:
                return self._mark_degraded("Workflow version 写入失败")
            return True
        except Exception as error:
            return self._mark_degraded(error)

    def _append_round(
        self,
        *,
        phase: str,
        batch_index: int,
        combination_index: int,
        round_record: Mapping[str, Any],
    ) -> bool:
        """读取、按轮次 upsert、重新整理并原子写回阶段 summary。"""

        if phase == "router":
            path = (
                f"05_router/{self._batch_path(batch_index)}/"
                f"{self._combination_path(combination_index)}/summary.json"
            )
        elif phase == "revision":
            path = (
                f"06_revision/{self._batch_path(batch_index)}/"
                f"{self._combination_path(combination_index)}/summary.json"
            )
        else:  # pragma: no cover - 仅由本类内部调用
            return self._mark_degraded("未知模型阶段")
        try:
            phase_directory = {
                "router": "05_router",
                "revision": "06_revision",
            }[phase]
            target_parent = (
                f"{phase_directory}/{self._batch_path(batch_index)}/"
                f"{self._combination_path(combination_index)}"
            )
            round_record = self._normalise_source_fields(
                round_record,
                target_parent,
            )
            existing = self.writer.read_model_output_view(path)
            if existing is None:
                if (
                    phase == "revision"
                    and round_record.get("round_index") != 1
                ):
                    raise ValueError("首个模型阶段 round_index 必须为 1")
                combined = {
                    "batch_index": batch_index,
                    "combination_index": combination_index,
                    "rounds": [deepcopy(dict(round_record))],
                }
                view = self._construct_round_summary(phase, combined)
                return self._write_summary(path, view)

            current = self._construct_round_summary(phase, existing)
            if (
                current["batch_index"] != batch_index
                or current["combination_index"] != combination_index
            ):
                raise ValueError("已有模型阶段 summary 的组合身份不一致")
            incoming_index = round_record.get("round_index")
            if isinstance(incoming_index, bool) or not isinstance(incoming_index, int):
                raise ValueError("模型阶段 round_index 必须是整数")
            last_index = current["rounds"][-1]["round_index"]
            if incoming_index == last_index:
                candidate = {
                    "batch_index": batch_index,
                    "combination_index": combination_index,
                    "rounds": [
                        *deepcopy(current["rounds"][:-1]),
                        deepcopy(dict(round_record)),
                    ],
                }
                candidate_view = self._construct_round_summary(phase, candidate)
                if candidate_view["rounds"][-1] == current["rounds"][-1]:
                    return True
                return self._mark_degraded(
                    "同一模型阶段 round_index 的内容发生冲突"
                )
            if incoming_index != last_index + 1:
                raise ValueError("新模型阶段 round_index 必须紧接已有最后一轮")
            combined = {
                "batch_index": batch_index,
                "combination_index": combination_index,
                "rounds": [
                    *deepcopy(current["rounds"]),
                    deepcopy(dict(round_record)),
                ],
            }
            view = self._construct_round_summary(phase, combined)
            return self._write_summary(path, view)
        except Exception as error:
            return self._mark_degraded(error)

    def append_router_round(
        self,
        batch_index: int,
        combination_index: int,
        round_record: Mapping[str, Any],
    ) -> bool:
        """追加或更新 Router 业务轮次；模型格式重答不由本方法递增轮次。"""

        return self._append_round(
            phase="router",
            batch_index=batch_index,
            combination_index=combination_index,
            round_record=round_record,
        )

    def append_revision_round(
        self,
        batch_index: int,
        combination_index: int,
        round_record: Mapping[str, Any],
    ) -> bool:
        """追加或更新 Revision 业务轮次，并保留能力缺口原文。"""

        return self._append_round(
            phase="revision",
            batch_index=batch_index,
            combination_index=combination_index,
            round_record=round_record,
        )


__all__ = ["ModelRunViewPublisher"]
