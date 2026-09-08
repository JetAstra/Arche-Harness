"""任务级 Review 证据快照/读取器的隔离、完整性和并发测试。"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chemistry_multiagent.investigation import evidence as evidence_module
from chemistry_multiagent.investigation.evidence import (
    EvidenceArtifactIntegrityError,
    EvidenceArtifactNotFound,
    EvidenceBudgetExceeded,
    EvidenceRegistryCorruption,
    EvidenceRequestError,
    ReviewEvidenceStore,
    UnsupportedEvidenceArtifact,
    read_snapshot_artifact,
)


RESULT_VECTOR = tuple((f"combination-{index}", "1") for index in range(1, 6))


def _empty_artifact_groups(result_vector=RESULT_VECTOR) -> dict[tuple[str, str], list[object]]:
    """为一个完整五项向量构造没有文件 artifact 的规范分组。"""

    return {pair: [] for pair in result_vector}


def _artifact_groups(artifact, result_vector=RESULT_VECTOR):
    """把一个 artifact 只绑定到第一个完整 pair，其余 pair 保持为空。"""

    groups = _empty_artifact_groups(result_vector)
    groups[result_vector[0]] = [artifact]
    return groups


def _register(store: ReviewEvidenceStore, source: Path, *, source_ref: str | None = None):
    """以真实 MiniChem-style source ref 和文件 SHA 登记 Evidence reader 来源。"""

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    selected_ref = source_ref or (
        "art_" + hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:32]
    )
    return store.register_source_artifact(
        source,
        source_artifact_ref=selected_ref,
        expected_source_sha256=digest,
    )


def test_snapshot_manifest_is_pair_scoped_path_free_and_idempotent(tmp_path: Path) -> None:
    """公开 manifest 只含四个顶层字段，重复规范化输入返回同一快照。"""

    source = tmp_path / "private" / "evidence.txt"
    source.parent.mkdir()
    source.write_text("第一段\n第二段\n", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "review-store")
    artifact = _register(store, source)

    first = store.create_batch_evidence_snapshot(
        "task-1",
        "batch-1",
        RESULT_VECTOR,
        _artifact_groups(artifact),
    )
    replay = store.create_batch_evidence_snapshot(
        "task-1",
        "batch-1",
        tuple(reversed(RESULT_VECTOR)),
        dict(reversed(tuple(_artifact_groups(artifact).items()))),
    )

    manifest = first.to_public_manifest()
    assert replay == first
    assert set(manifest) == {"snapshot_id", "task_id", "batch_id", "result_pairs"}
    assert len(manifest["result_pairs"]) == 5
    assert manifest["result_pairs"][0]["artifacts"] == [artifact.to_public_dict()]
    assert all(not pair["artifacts"] for pair in manifest["result_pairs"][1:])
    assert str(source) not in json.dumps(manifest, ensure_ascii=False)
    assert "source_path" not in manifest
    assert oct(os.stat(store.root_dir).st_mode & 0o777) == "0o700"
    assert oct(os.stat(store.root_dir / "registry.json").st_mode & 0o777) == "0o600"
    assert oct(os.stat(store.root_dir / "registry.lock").st_mode & 0o777) == "0o600"


def test_empty_pairs_and_empty_snapshot_are_valid_but_group_shape_is_strict(tmp_path: Path) -> None:
    """没有文件 artifact 仍可审查，多余/缺失/路径/版本漂移输入必须失败。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    empty = store.create_batch_evidence_snapshot(
        "task",
        "batch",
        RESULT_VECTOR,
        _empty_artifact_groups(),
    )
    assert all(not pair.artifacts for pair in empty.result_pairs)

    extra = _empty_artifact_groups()
    extra[("combination-6", "1")] = []
    with pytest.raises(EvidenceRequestError, match="完整覆盖"):
        store.create_batch_evidence_snapshot("task", "batch", RESULT_VECTOR, extra)

    missing = _empty_artifact_groups()
    missing.pop(RESULT_VECTOR[0])
    with pytest.raises(EvidenceRequestError, match="完整覆盖"):
        store.create_batch_evidence_snapshot("task", "batch", RESULT_VECTOR, missing)

    with pytest.raises(EvidenceRequestError, match="key"):
        store.create_batch_evidence_snapshot("task", "batch", RESULT_VECTOR, {"combination-1": []})

    drifted_vector = (*RESULT_VECTOR[:1], ("combination-2", "2"), *RESULT_VECTOR[2:])
    with pytest.raises(EvidenceRequestError, match="完整覆盖"):
        store.create_batch_evidence_snapshot(
            "task",
            "batch",
            drifted_vector,
            _empty_artifact_groups(),
        )

    source = tmp_path / "evidence.txt"
    source.write_text("evidence", encoding="utf-8")
    artifact = _register(store, source)
    with pytest.raises(EvidenceRequestError, match="opaque ID"):
        store.create_batch_evidence_snapshot(
            "task",
            "batch-2",
            RESULT_VECTOR,
            _artifact_groups(source),
        )
    with pytest.raises(EvidenceRequestError, match="opaque ID"):
        store.create_batch_evidence_snapshot(
            "task",
            "batch-3",
            RESULT_VECTOR,
            _artifact_groups(artifact.to_public_dict()),
        )

    reused = _empty_artifact_groups()
    reused[RESULT_VECTOR[0]] = [artifact]
    reused[RESULT_VECTOR[1]] = [artifact]
    with pytest.raises(EvidenceRequestError, match="跨 pair"):
        store.create_batch_evidence_snapshot("task", "batch-4", RESULT_VECTOR, reused)


def test_snapshot_accepts_author_batch_with_three_result_pairs(tmp_path: Path) -> None:
    """作者入口的三条当前结果可以形成一个完整、不可变的 Review 快照。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    result_vector = RESULT_VECTOR[:3]

    snapshot = store.create_batch_evidence_snapshot(
        "task",
        "author-batch",
        result_vector,
        _empty_artifact_groups(result_vector),
    )

    assert tuple(
        (pair.combination_id, pair.result_version)
        for pair in snapshot.result_pairs
    ) == result_vector


def test_default_snapshot_artifact_limit_accepts_64_and_rejects_65(tmp_path: Path) -> None:
    """默认快照允许 64 个已核对产物，第 65 个仍由数量边界拒绝。"""

    result_vector = (("combination-1", "1"),)
    store = ReviewEvidenceStore(tmp_path / "store")
    artifacts = []
    for index in range(65):
        source = tmp_path / f"artifact-{index:02d}.txt"
        source.write_text(f"artifact-{index}\n", encoding="utf-8")
        artifacts.append(_register(store, source))

    snapshot = store.create_batch_evidence_snapshot(
        "task",
        "batch-64",
        result_vector,
        {result_vector[0]: artifacts[:64]},
    )
    assert len(snapshot.result_pairs[0].artifacts) == 64

    with pytest.raises(EvidenceRequestError, match="快照 artifact 数量超过上限"):
        store.create_batch_evidence_snapshot(
            "task",
            "batch-65",
            result_vector,
            {result_vector[0]: artifacts},
        )


def test_snapshot_isolation_and_only_read_adapter_are_enforced(tmp_path: Path) -> None:
    """artifact 只能从所属快照读取，唯一顶层 adapter 不能注册或接收路径。"""

    first_source = tmp_path / "first.txt"
    second_source = tmp_path / "second.txt"
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    first_artifact = _register(store, first_source)
    second_artifact = _register(store, second_source)
    first = store.create_batch_evidence_snapshot(
        "task", "batch-1", RESULT_VECTOR, _artifact_groups(first_artifact)
    )
    second = store.create_batch_evidence_snapshot(
        "task", "batch-2", RESULT_VECTOR, _artifact_groups(second_artifact)
    )

    with pytest.raises(EvidenceArtifactNotFound):
        store.read_snapshot_artifact(first.snapshot_id, second_artifact.artifact_id)
    with pytest.raises(EvidenceRequestError):
        store.read_snapshot_artifact(first.snapshot_id, str(second_source))
    with pytest.raises(EvidenceRequestError):
        store.read_snapshot_artifact(str(first_source), first_artifact.artifact_id)
    assert store.read_snapshot_artifact(second.snapshot_id, second_artifact.artifact_id).content == "second"

    adapted = read_snapshot_artifact(store, second.snapshot_id, second_artifact.artifact_id)
    assert adapted["content"] == "second"
    assert "register_evidence_artifact" not in evidence_module.__all__
    assert "create_batch_evidence_snapshot" not in evidence_module.__all__
    assert "read_batch_evidence_artifact" not in evidence_module.__all__
    with pytest.raises(EvidenceRequestError):
        read_snapshot_artifact(store, second.snapshot_id, str(second_source))


def test_read_rechecks_size_and_hash_without_writing_source(tmp_path: Path) -> None:
    """来源篡改后读取失败，读取不会将新内容覆盖回来源。"""

    source = tmp_path / "tamper.txt"
    source.write_text("original", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    artifact = _register(store, source)
    snapshot = store.create_batch_evidence_snapshot(
        "task", "batch", RESULT_VECTOR, _artifact_groups(artifact)
    )

    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(EvidenceArtifactIntegrityError):
        store.read_snapshot_artifact(snapshot.snapshot_id, artifact.artifact_id)
    assert source.read_text(encoding="utf-8") == "tampered"


def test_read_paginates_and_budget_is_shared_by_two_store_instances(tmp_path: Path) -> None:
    """分页、单次上限和同一 reader 进程内跨 store 累计预算均生效。"""

    source = tmp_path / "pages.txt"
    source.write_text("abcdefghij", encoding="utf-8")
    root = tmp_path / "store"
    first_store = ReviewEvidenceStore(root, max_single_read_chars=4, max_total_read_chars=7)
    artifact = _register(first_store, source)
    snapshot = first_store.create_batch_evidence_snapshot(
        "task", "batch", RESULT_VECTOR, _artifact_groups(artifact)
    )
    second_store = ReviewEvidenceStore(root, max_single_read_chars=4, max_total_read_chars=7)

    with pytest.raises(EvidenceRequestError):
        first_store.read_snapshot_artifact(snapshot.snapshot_id, artifact.artifact_id, 0, 5)
    first = first_store.read_snapshot_artifact(snapshot.snapshot_id, artifact.artifact_id, 0, 4)
    second = second_store.read_snapshot_artifact(
        snapshot.snapshot_id,
        artifact.artifact_id,
        first.next_offset or 0,
        4,
    )
    assert first.content == "abcd"
    assert first.truncated is True and first.next_offset == 4
    assert second.content == "efg"
    assert second.truncated is True and second.budget_remaining == 0
    with pytest.raises(EvidenceBudgetExceeded):
        first_store.read_snapshot_artifact(snapshot.snapshot_id, artifact.artifact_id, 7, 1)


def test_json_pointer_and_binary_unknown_limits_are_controlled(tmp_path: Path) -> None:
    """JSON pointer 只导航 JSON，image/unknown 文件在登记阶段受控拒绝。"""

    source = tmp_path / "evidence.json"
    source.write_text('{"facts": ["alpha", "beta"], "nested": {"answer": 42}}', encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    artifact = _register(store, source)
    snapshot = store.create_batch_evidence_snapshot(
        "task", "batch", RESULT_VECTOR, _artifact_groups(artifact)
    )
    selected = store.read_snapshot_artifact(
        snapshot.snapshot_id,
        artifact.artifact_id,
        0,
        10,
        "/nested/answer",
    )
    assert selected.content == "42"
    with pytest.raises(EvidenceRequestError):
        store.read_snapshot_artifact(
            snapshot.snapshot_id,
            artifact.artifact_id,
            0,
            2,
            "/../secret",
        )

    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(UnsupportedEvidenceArtifact):
        _register(store, image)
    unknown = tmp_path / "evidence.unknown"
    unknown.write_text("text", encoding="utf-8")
    with pytest.raises(UnsupportedEvidenceArtifact):
        _register(store, unknown)


def test_two_store_instances_register_without_losing_registry_updates(tmp_path: Path) -> None:
    """两个并发 store 实例写同一 registry 时，POSIX 文件锁保留双方登记。"""

    root = tmp_path / "shared-store"
    sources = []
    for index in range(2):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"content-{index}", encoding="utf-8")
        sources.append(source)

    def register(path: Path) -> str:
        return _register(ReviewEvidenceStore(root), path).artifact_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        artifact_ids = list(executor.map(register, sources))

    registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    assert set(artifact_ids) == set(registry["artifacts"])
    assert len(registry["artifacts"]) == 2


def test_public_manifest_rejects_snapshot_content_or_artifact_registry_tampering(
    tmp_path: Path,
) -> None:
    """公开 manifest 恢复时同时核对 snapshot ID 和每个 artifact 注册摘要。"""

    source = tmp_path / "evidence.txt"
    source.write_text("evidence", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    artifact = _register(store, source)
    snapshot = store.create_batch_evidence_snapshot(
        "task", "batch", RESULT_VECTOR, _artifact_groups(artifact)
    )
    registry_path = store.root_dir / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["snapshots"][snapshot.snapshot_id]["batch_id"] = "batch-tampered"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvidenceRegistryCorruption, match="snapshot_id"):
        store.read_public_snapshot_manifest(snapshot.snapshot_id)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["snapshots"][snapshot.snapshot_id]["batch_id"] = "batch"
    pair_artifact = registry["snapshots"][snapshot.snapshot_id]["result_pairs"][0]["artifacts"][0]
    pair_artifact["size"] += 1
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvidenceRegistryCorruption, match="artifact"):
        store.read_public_snapshot_manifest(snapshot.snapshot_id)


@pytest.mark.parametrize("reference_kind", ["manifest", "id"])
def test_create_snapshot_rechecks_registered_source_before_freezing(
    tmp_path: Path,
    reference_kind: str,
) -> None:
    """来源在登记后被替换时，建快照必须拒绝 manifest 和 opaque ID 两种引用。"""

    source = tmp_path / "evidence.txt"
    source.write_text("before", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    artifact = _register(store, source)

    # 保持文件大小不变，确保测试命中 SHA-256 重验而非只命中 size 检查。
    source.write_text("change", encoding="utf-8")
    reference = artifact if reference_kind == "manifest" else artifact.artifact_id
    with pytest.raises(EvidenceArtifactIntegrityError):
        store.create_batch_evidence_snapshot(
            "task", "batch", RESULT_VECTOR, _artifact_groups(reference)
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "/private/task"),
        ("batch_id", "/private/batch"),
        ("combination_id", "/private/combination"),
        ("result_version", ""),
    ],
)
def test_registry_task_batch_and_pair_field_errors_become_corruption(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    """registry 内部字段即使触发公开文本校验，也统一报告为 corruption。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    snapshot = store.create_batch_evidence_snapshot(
        "task", "batch", RESULT_VECTOR, _empty_artifact_groups()
    )
    registry_path = store.root_dir / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    raw_snapshot = registry["snapshots"][snapshot.snapshot_id]
    if field in {"task_id", "batch_id"}:
        raw_snapshot[field] = value
    else:
        raw_snapshot["result_pairs"][0][field] = value
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(EvidenceRegistryCorruption):
        store.read_public_snapshot_manifest(snapshot.snapshot_id)


def test_registration_requires_authenticated_source_binding_and_exact_file_sha256(
    tmp_path: Path,
) -> None:
    """登记必须同时有外层 source ref 和实际 UTF-8 文件摘要，且 reader ID 分离。"""

    source = tmp_path / "binding.txt"
    source.write_text("source bytes", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source_ref = "art_" + "1" * 32
    store = ReviewEvidenceStore(tmp_path / "store")

    with pytest.raises(TypeError):
        store.register_source_artifact(source)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        store.register_source_artifact(  # type: ignore[call-arg]
            source,
            source_artifact_ref=source_ref,
        )
    with pytest.raises(EvidenceRequestError):
        store.register_source_artifact(
            source,
            source_artifact_ref="artifact-without-minichem-shape",
            expected_source_sha256=digest,
        )
    with pytest.raises(EvidenceArtifactIntegrityError):
        store.register_source_artifact(
            source,
            source_artifact_ref=source_ref,
            expected_source_sha256="0" * 64,
        )

    manifest = store.register_source_artifact(
        source,
        source_artifact_ref=source_ref,
        expected_source_sha256=digest,
    )
    assert manifest.source_artifact_ref == source_ref
    assert manifest.sha256 == digest
    assert manifest.artifact_id != source_ref
    assert str(source) not in json.dumps(manifest.to_public_dict(), ensure_ascii=False)


def test_registration_accepts_utf8_sdf_as_review_text_evidence(tmp_path: Path) -> None:
    """SDF 是可审查的 UTF-8 化学文本，不能因媒体类型被误判为二进制。"""

    source = tmp_path / "ethanol.sdf"
    source.write_text("ethanol\n  MiniChem\n\n$$$$\n", encoding="utf-8")
    manifest = _register(ReviewEvidenceStore(tmp_path / "store"), source)

    assert manifest.media_type == "chemical/x-mdl-sdfile"


def test_source_ref_is_part_of_evidence_id_derivation(tmp_path: Path) -> None:
    """同一路径和内容换用不同 MiniChem source ref 时不能复用 reader ID。"""

    source = tmp_path / "same-content.txt"
    source.write_text("same bytes", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    store = ReviewEvidenceStore(tmp_path / "store")
    first = store.register_source_artifact(
        source,
        source_artifact_ref="art_" + "2" * 32,
        expected_source_sha256=digest,
    )
    second = store.register_source_artifact(
        source,
        source_artifact_ref="art_" + "3" * 32,
        expected_source_sha256=digest,
    )
    assert first.artifact_id != second.artifact_id
    assert first.source_artifact_ref != second.source_artifact_ref


def test_snapshot_rejects_source_binding_reuse_and_registry_source_drift(
    tmp_path: Path,
) -> None:
    """source ref 跨 pair 复用或快照登记后发生绑定漂移都 fail-closed。"""

    first_source = tmp_path / "first.txt"
    second_source = tmp_path / "second.txt"
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    shared_ref = "art_" + "4" * 32
    first = _register(store, first_source, source_ref=shared_ref)
    second = _register(store, second_source, source_ref=shared_ref)
    groups = _empty_artifact_groups()
    groups[RESULT_VECTOR[0]] = [first]
    groups[RESULT_VECTOR[1]] = [second]
    with pytest.raises(EvidenceRequestError, match="source artifact ref"):
        store.create_batch_evidence_snapshot("task", "batch", RESULT_VECTOR, groups)

    independent = _register(store, second_source, source_ref="art_" + "5" * 32)
    snapshot = store.create_batch_evidence_snapshot(
        "task", "batch-2", RESULT_VECTOR, _artifact_groups(independent)
    )
    registry_path = store.root_dir / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["snapshots"][snapshot.snapshot_id]["result_pairs"][0]["artifacts"][0][
        "source_artifact_ref"
    ] = "art_" + "6" * 32
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvidenceRegistryCorruption, match="artifact"):
        store.read_public_snapshot_manifest(snapshot.snapshot_id)


def test_schema_v1_cannot_be_silently_upgraded(tmp_path: Path) -> None:
    """旧 schema 缺少 source binding 时必须明确拒绝，不能猜测迁移。"""

    root = tmp_path / "store"
    store = ReviewEvidenceStore(root)
    registry_path = root / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["schema_version"] = 1
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvidenceRegistryCorruption, match="schema v1"):
        ReviewEvidenceStore(root)


def test_registered_artifact_loader_rejects_registry_field_and_binding_drift(
    tmp_path: Path,
) -> None:
    """登记项字段、来源摘要和派生 ID 任一漂移都必须转换为 corruption。"""

    source = tmp_path / "registered.txt"
    source.write_text("registered", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    manifest = _register(store, source)
    registry = store._load_registry()
    raw = registry["artifacts"][manifest.artifact_id]
    invalid_mutations = (
        lambda item: item.pop("sha256"),
        lambda item: item.__setitem__("size", True),
        lambda item: item.__setitem__("source_artifact_ref", "not-an-artifact"),
        lambda item: item.__setitem__("source_path", "relative/path"),
        lambda item: item.__setitem__("source_path_digest", "0" * 64),
        lambda item: item.__setitem__("media_type", "application/octet-stream"),
        lambda item: item.__setitem__("artifact_id", "wrong-id"),
    )
    for mutate in invalid_mutations:
        candidate = json.loads(json.dumps(raw))
        mutate(candidate)
        broken_registry = {"artifacts": {manifest.artifact_id: candidate}}
        with pytest.raises(EvidenceRegistryCorruption):
            store._load_registered_artifact(broken_registry, manifest.artifact_id)


def test_json_pointer_validation_navigation_and_serialization_guards(
    tmp_path: Path,
) -> None:
    """JSON pointer 支持 RFC 转义和数组导航，并拒绝越界、标量和非 JSON。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    assert store._validate_json_pointer(None) is None
    assert store._validate_json_pointer("") == ""
    assert store._validate_json_pointer("/a~1b/~0key") == "/a~1b/~0key"
    invalid_pointers = (123, "plain", "\x00", "/~2", "/..", "/" + "a" * 513)
    for pointer in invalid_pointers:
        with pytest.raises(EvidenceRequestError):
            store._validate_json_pointer(pointer)

    source_text = json.dumps(
        {"a/b": {"~key": "value"}, "items": ["first", {"answer": 42}], "scalar": 1},
        ensure_ascii=False,
    )
    assert store._select_json_pointer_text(source_text, "application/json", "") == json.dumps(
        json.loads(source_text), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert store._select_json_pointer_text(source_text, "application/json", "/a~1b/~0key") == '"value"'
    assert store._select_json_pointer_text(source_text, "application/json", "/items/0") == '"first"'
    assert store._select_json_pointer_text(source_text, "application/json", "/items/1/answer") == "42"

    invalid_selections = (
        ("/items/01", "json_pointer"),
        ("/items/9", "json_pointer"),
        ("/items/nope", "json_pointer"),
        ("/missing", "json_pointer"),
        ("/scalar/child", "json_pointer"),
        ("/items/0", "text/plain"),
    )
    for pointer, media_type in invalid_selections:
        with pytest.raises(EvidenceRequestError):
            store._select_json_pointer_text(source_text, media_type, pointer)
    with pytest.raises(EvidenceRequestError):
        store._select_json_pointer_text("not-json", "application/json", "/items")


def test_snapshot_manifest_artifact_fields_fail_closed(tmp_path: Path) -> None:
    """快照 registry 中的 artifact manifest 字段必须逐项满足公开摘要合同。"""

    source = tmp_path / "manifest.txt"
    source.write_text("manifest", encoding="utf-8")
    store = ReviewEvidenceStore(tmp_path / "store")
    manifest = _register(store, source)
    valid = manifest.to_public_dict()
    invalid_values = (
        None,
        {**valid, "artifact_id": "bad"},
        {**valid, "source_artifact_ref": "bad"},
        {**valid, "size": True},
        {**valid, "sha256": "bad"},
        {**valid, "media_type": "application/octet-stream"},
    )
    for value in invalid_values:
        with pytest.raises(EvidenceRegistryCorruption):
            evidence_module._build_snapshot_artifact_manifest(store, value)
