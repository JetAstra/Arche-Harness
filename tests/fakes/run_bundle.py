"""测试专用的 schema-v10 历史 Bundle fixture 构造器。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from chemistry_multiagent.utils.run_bundle import (
    INVESTIGATION_STAGE_LAYOUT,
    INVESTIGATION_STAGE_NAMES,
    LEGACY_STAGE_LAYOUT,
    LEGACY_STAGE_NAMES,
    RunBundle,
)


class _SchemaV10BundleFixtureWriter:
    """只在测试中生成旧 ``outputs/runs`` Bundle，不属于生产入口。"""

    @staticmethod
    def create(
        work_dir: str | Path,
        run_id: str,
        *,
        stage_layout: str = LEGACY_STAGE_LAYOUT,
        scientific_question: str | None = None,
        task_config: Mapping[str, Any] | None = None,
        model_configuration: Mapping[str, Any] | None = None,
    ) -> RunBundle:
        """通过正式 Bundle 的固定文件生成历史 schema-v10 fixture。"""

        if stage_layout not in {
            LEGACY_STAGE_LAYOUT,
            INVESTIGATION_STAGE_LAYOUT,
        }:
            raise ValueError("schema-v10 fixture 阶段布局只能使用历史 Bundle 布局")

        workspace = Path(work_dir).expanduser().resolve(strict=False)
        legacy_namespace = workspace / "outputs" / "runs"
        legacy_namespace.mkdir(parents=True, exist_ok=True)
        legacy_root = legacy_namespace / run_id
        with tempfile.TemporaryDirectory(
            prefix=".schema-v10-fixture-",
            dir=str(workspace),
        ) as transient_root:
            formal = RunBundle(
                transient_root,
                research_task_id=run_id,
                scientific_question=scientific_question,
                task_config=task_config,
                model_configuration=model_configuration,
            )
            formal_stage_names = tuple(formal.stage_names)
            formal.root.rename(legacy_root)
        (workspace / "outputs" / ".index.lock").touch()

        if stage_layout == INVESTIGATION_STAGE_LAYOUT:
            stage_mapping = dict(
                zip(
                    formal_stage_names,
                    INVESTIGATION_STAGE_NAMES,
                    strict=True,
                )
            )
        else:
            stage_mapping = {
                formal_stage_names[0]: LEGACY_STAGE_NAMES[0],
                formal_stage_names[1]: LEGACY_STAGE_NAMES[1],
                formal_stage_names[2]: LEGACY_STAGE_NAMES[5],
                formal_stage_names[3]: LEGACY_STAGE_NAMES[-1],
            }

        stage_root = legacy_root / "stages"
        stage_documents: dict[str, dict[str, Any]] = {}
        for source_name in formal_stage_names:
            source_path = stage_root / f"{source_name}.json"
            stage_documents[source_name] = json.loads(
                source_path.read_text(encoding="utf-8")
            )
            source_path.unlink()

        mapped_stage_documents: dict[str, dict[str, Any]] = {}
        for source_name, stage_name in stage_mapping.items():
            stage_document = stage_documents[source_name]
            stage_document["stage"] = stage_name
            stage_document["run_id"] = stage_document.pop("research_task_id")
            stage_document["recovery"] = {
                "available": False,
                "waiting_executions": [],
                "pauses": [],
            }
            mapped_stage_documents[stage_name] = stage_document

        if stage_layout == LEGACY_STAGE_LAYOUT:
            template = mapped_stage_documents[LEGACY_STAGE_NAMES[0]]
            for stage_name in LEGACY_STAGE_NAMES:
                if stage_name not in mapped_stage_documents:
                    mapped_stage_documents[stage_name] = {
                        "schema_version": template["schema_version"],
                        "run_id": run_id,
                        "stage": stage_name,
                        "status": "not_started",
                        "started_at": None,
                        "finished_at": None,
                        "summary": {},
                        "entries": [],
                        "artifacts": [],
                        "blocking": None,
                        "recovery": {
                            "available": False,
                            "waiting_executions": [],
                            "pauses": [],
                        },
                    }

        for stage_name in (
            INVESTIGATION_STAGE_NAMES
            if stage_layout == INVESTIGATION_STAGE_LAYOUT
            else LEGACY_STAGE_NAMES
        ):
            stage_root.joinpath(f"{stage_name}.json").write_text(
                json.dumps(
                    mapped_stage_documents[stage_name],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        for document_name in ("run.json", "status.json", "summary.json"):
            path = legacy_root / document_name
            document = json.loads(path.read_text(encoding="utf-8"))
            document["run_id"] = document.pop("research_task_id")
            document["recovery"] = {
                "available": False,
                "waiting_executions": [],
                "pauses": [],
            }
            if document_name == "run.json":
                document["stage_layout"] = stage_layout
                document["stage_order"] = list(
                    INVESTIGATION_STAGE_NAMES
                    if stage_layout == INVESTIGATION_STAGE_LAYOUT
                    else LEGACY_STAGE_NAMES
                )
                document["paths"] = {
                    **document.get("paths", {}),
                    "run_root": f"outputs/runs/{run_id}",
                }
                document["paths"].pop("research_task_root", None)
            elif document_name == "summary.json":
                document["stage_order"] = list(
                    INVESTIGATION_STAGE_NAMES
                    if stage_layout == INVESTIGATION_STAGE_LAYOUT
                    else LEGACY_STAGE_NAMES
                )
                stage_records = document.get("stages", {})
                document["stages"] = {
                    stage_name: stage_records.get(
                        source_name,
                        {
                            "status": "not_started",
                            "path": f"stages/{stage_name}.json",
                            "entry_count": 0,
                            "status_counts": {},
                        },
                    )
                    for source_name, stage_name in stage_mapping.items()
                }
                if stage_layout == LEGACY_STAGE_LAYOUT:
                    for stage_name in LEGACY_STAGE_NAMES:
                        document["stages"].setdefault(
                            stage_name,
                            {
                                "status": "not_started",
                                "path": f"stages/{stage_name}.json",
                                "entry_count": 0,
                                "status_counts": {},
                            },
                        )
            path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        (stage_root / "README.md").write_text(
            "# 阶段摘要\n\n" + "\n".join(
                f"- `{stage_name}.json`" for stage_name in (
                    INVESTIGATION_STAGE_NAMES
                    if stage_layout == INVESTIGATION_STAGE_LAYOUT
                    else LEGACY_STAGE_NAMES
                )
            ) + "\n",
            encoding="utf-8",
        )
        (legacy_namespace / "README.md").write_text(
            "# 运行 Bundle\n\n历史 schema-v10 测试 fixture。\n",
            encoding="utf-8",
        )
        return RunBundle.open_existing_legacy(workspace, run_id)
