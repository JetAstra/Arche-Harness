"""验证 Run Bundle 的阶段布局冻结、恢复和 fail-closed 校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemistry_multiagent.utils.run_bundle import (
    INVESTIGATION_STAGE_LAYOUT,
    INVESTIGATION_STAGE_NAMES,
    LEGACY_STAGE_LAYOUT,
    LEGACY_STAGE_NAMES,
    RunBundle,
)
from tests.fakes.run_bundle import _SchemaV10BundleFixtureWriter


def _read_json(path: Path) -> dict:
    """读取测试用 JSON 对象。"""

    return json.loads(path.read_text(encoding="utf-8"))


def test_schema_v10_investigation_layout_is_read_only_in_recovery(
    tmp_path: Path,
) -> None:
    """schema-v10 历史 Investigation Bundle 只读恢复并保持冻结的四阶段布局。"""

    bundle = _SchemaV10BundleFixtureWriter.create(
        tmp_path,
        "investigation-layout",
        stage_layout=INVESTIGATION_STAGE_LAYOUT,
        scientific_question="验证调查布局",
    )

    assert bundle.stage_layout == INVESTIGATION_STAGE_LAYOUT
    assert bundle.stage_names == INVESTIGATION_STAGE_NAMES
    with pytest.raises(AttributeError):
        bundle.stage_names = LEGACY_STAGE_NAMES  # type: ignore[misc]
    run_document = _read_json(bundle.run_path)
    summary_document = _read_json(bundle.summary_path)
    assert run_document["stage_layout"] == INVESTIGATION_STAGE_LAYOUT
    assert run_document["stage_order"] == list(INVESTIGATION_STAGE_NAMES)
    assert summary_document["stage_order"] == list(INVESTIGATION_STAGE_NAMES)
    assert {
        path.name.removesuffix(".json")
        for path in (bundle.root / "stages").glob("*.json")
    } == set(INVESTIGATION_STAGE_NAMES)
    stages_readme = (bundle.root / "stages" / "README.md").read_text(encoding="utf-8")
    assert "investigation_and_experiment.json" in stages_readme
    assert "admission_debate.json" not in stages_readme

    with pytest.raises(RuntimeError, match="历史 Bundle 只读"):
        bundle.write_stage_summary(
            "investigation_and_experiment",
            {"status": "completed", "observation_count": 1},
        )
    with pytest.raises(RuntimeError, match="历史 Bundle 只读"):
        bundle.write_stage_summary("planning", {"status": "completed"})
    with pytest.raises(ValueError, match="阶段布局"):
        _SchemaV10BundleFixtureWriter.create(
            tmp_path,
            "investigation-layout-invalid",
            stage_layout="unknown",
        )

    resumed = RunBundle.open_existing_legacy(tmp_path, "investigation-layout")
    assert resumed.stage_layout == INVESTIGATION_STAGE_LAYOUT
    assert resumed.stage_names == INVESTIGATION_STAGE_NAMES
    opened = RunBundle.open_existing_legacy(tmp_path, "investigation-layout")
    assert opened.stage_layout == INVESTIGATION_STAGE_LAYOUT
    assert opened.stage_names == INVESTIGATION_STAGE_NAMES


def test_schema_v1_bundle_without_layout_fields_still_infers_legacy(
    tmp_path: Path,
) -> None:
    """没有布局字段的 schema v1 九阶段 Bundle 只能恢复为 legacy。"""

    bundle = _SchemaV10BundleFixtureWriter.create(
        tmp_path,
        "legacy-layout",
        stage_layout=LEGACY_STAGE_LAYOUT,
    )
    run_document = _read_json(bundle.run_path)
    run_document.pop("stage_layout")
    run_document.pop("stage_order")
    bundle.run_path.write_text(
        json.dumps(run_document, ensure_ascii=False), encoding="utf-8"
    )

    opened = RunBundle.open_existing_legacy(tmp_path, "legacy-layout")
    assert opened.stage_layout == LEGACY_STAGE_LAYOUT
    assert opened.stage_names == LEGACY_STAGE_NAMES
    assert _read_json(opened.run_path).get("stage_layout") is None


@pytest.mark.parametrize("document_name", ["run", "summary"])
def test_legacy_stage_order_drift_is_rejected(
    tmp_path: Path,
    document_name: str,
) -> None:
    """run.json 与 summary.json 的阶段顺序漂移都必须拒绝恢复。"""

    bundle = _SchemaV10BundleFixtureWriter.create(
        tmp_path,
        f"stage-order-{document_name}",
        stage_layout=INVESTIGATION_STAGE_LAYOUT,
    )
    path = bundle.run_path if document_name == "run" else bundle.summary_path
    document = _read_json(path)
    document["stage_order"] = list(LEGACY_STAGE_NAMES)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="stage_order"):
        RunBundle.open_existing_legacy(tmp_path, f"stage-order-{document_name}")


def test_unknown_layout_and_nonlegacy_layoutless_bundle_are_rejected(
    tmp_path: Path,
) -> None:
    """非法布局以及无布局字段的非 legacy 阶段顺序必须 fail-closed。"""

    with pytest.raises(ValueError, match="阶段布局"):
        _SchemaV10BundleFixtureWriter.create(
            tmp_path,
            "invalid-layout",
            stage_layout="unknown",
        )

    bundle = _SchemaV10BundleFixtureWriter.create(
        tmp_path,
        "layoutless-investigation",
        stage_layout=LEGACY_STAGE_LAYOUT,
    )
    run_document = _read_json(bundle.run_path)
    run_document.pop("stage_layout")
    run_document["stage_order"] = list(INVESTIGATION_STAGE_NAMES)
    bundle.run_path.write_text(
        json.dumps(run_document, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="legacy|stage_order"):
        RunBundle.open_existing_legacy(tmp_path, "layoutless-investigation")


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_legacy_stage_file_set_must_match_frozen_layout(
    tmp_path: Path,
    mutation: str,
) -> None:
    """阶段目录缺文件或多文件时不得恢复，也不得自动迁移。"""

    bundle = _SchemaV10BundleFixtureWriter.create(
        tmp_path,
        f"stage-files-{mutation}",
        stage_layout=INVESTIGATION_STAGE_LAYOUT,
    )
    if mutation == "extra":
        (bundle.root / "stages" / "planning.json").write_text("{}", encoding="utf-8")
    else:
        (bundle.root / "stages" / "hypothesis.json").unlink()

    with pytest.raises(ValueError, match="阶段"):
        RunBundle.open_existing_legacy(tmp_path, f"stage-files-{mutation}")
    if mutation == "extra":
        assert (bundle.root / "stages" / "planning.json").is_file()
