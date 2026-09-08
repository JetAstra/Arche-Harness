"""service 侧逐文件 ArtifactRef 结果指针合同测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from minichem_toolbox import service


class _Reservation:
    """隔离 service 测试所需的最小资源预留对象。"""

    resource_allocation = {
        "cpu_ids": [0],
        "gpu_ids": [],
        "cpu_cores": 1,
        "memory_mb": 4096,
        "gpu_count": 0,
    }

    def release(self) -> None:
        """测试预留不需要释放外部资源。"""


def _stub_service_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    worker: dict,
) -> dict:
    """用固定 worker 响应调用公开 service.execute_action。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        service,
        "probe_all_backends",
        lambda specifications: {
            specification.id: {"available": True, "status": "available"}
            for specification in specifications
        },
    )
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "test-catalog-hash")
    monkeypatch.setattr(
        service,
        "reserve_resources",
        lambda *_args, **_kwargs: _Reservation(),
    )
    monkeypatch.setattr(service, "invoke_worker", lambda **_kwargs: dict(worker))
    return service.execute_action(
        "smiles2sdf",
        {
            "inputs": {"smiles_list": ["O"]},
            "resource_limits": {"cpu_cores": 1, "memory_mb": 4096},
        },
    )


def test_service_injects_compact_artifact_ref_at_existing_result_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """注册成员文件后，公开结果和 ActionResult JSON 都应包含紧凑引用。"""

    (tmp_path / "result.sdf").write_text("primary\n", encoding="utf-8")
    (tmp_path / "record-0.sdf").write_text("record\n", encoding="utf-8")
    response = _stub_service_dispatch(
        monkeypatch,
        tmp_path,
        {
            "status": "success",
            "result": {
                "successful_records": [
                    {"item_id": "smiles_0000", "sdf_artifact": None}
                ]
            },
            "artifact_files": [
                {
                    "path": "result.sdf",
                    "semantic_type": "SDFFile",
                    "media_type": "chemical/x-mdl-sd",
                },
                {
                    "path": "record-0.sdf",
                    "semantic_type": "SDFRecordFile",
                    "media_type": "chemical/x-mdl-sd",
                    "result_pointer": "/successful_records/0/sdf_artifact",
                },
            ],
        },
    )

    assert response["status"] == "success"
    compact_ref = response["result"]["successful_records"][0]["sdf_artifact"]
    assert set(compact_ref) == {"artifact_id"}
    member = next(
        item
        for item in response["output_artifacts"]
        if item["semantic_type"] == "SDFRecordFile"
    )
    assert compact_ref == {"artifact_id": member["artifact_id"]}

    result_artifact = next(
        item
        for item in response["output_artifacts"]
        if item["semantic_type"] == "ActionResult"
    )
    persisted = service.ArtifactStore().load(result_artifact)
    assert persisted["successful_records"][0]["sdf_artifact"] == compact_ref


@pytest.mark.parametrize(
    "result_pointer",
    [
        "successful_records/0/sdf_artifact",
        "/successful_records/0/sdf_artifact~2",
    ],
)
def test_service_rejects_malformed_result_pointer_structurally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result_pointer: str,
) -> None:
    """畸形 JSON Pointer 必须在登记文件前转换为稳定错误。"""

    (tmp_path / "result.sdf").write_text("primary\n", encoding="utf-8")
    (tmp_path / "record-0.sdf").write_text("record\n", encoding="utf-8")
    response = _stub_service_dispatch(
        monkeypatch,
        tmp_path,
        {
            "status": "success",
            "result": {
                "successful_records": [
                    {"item_id": "smiles_0000", "sdf_artifact": None}
                ]
            },
            "artifact_files": [
                {
                    "path": "result.sdf",
                    "semantic_type": "SDFFile",
                    "media_type": "chemical/x-mdl-sd",
                },
                {
                    "path": "record-0.sdf",
                    "semantic_type": "SDFRecordFile",
                    "media_type": "chemical/x-mdl-sd",
                    "result_pointer": result_pointer,
                },
            ],
        },
    )

    assert response["status"] == "failed"
    assert response["error"]["code"] == "result_pointer_malformed"
    assert not any(
        item["semantic_type"] in {"SDFFile", "SDFRecordFile"}
        for item in response["output_artifacts"]
    )


def test_service_rejects_duplicate_result_pointers_structurally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """两个文件不能声明同一个结果写入位置。"""

    (tmp_path / "result.sdf").write_text("primary\n", encoding="utf-8")
    (tmp_path / "record-0.sdf").write_text("record 0\n", encoding="utf-8")
    (tmp_path / "record-1.sdf").write_text("record 1\n", encoding="utf-8")
    response = _stub_service_dispatch(
        monkeypatch,
        tmp_path,
        {
            "status": "success",
            "result": {
                "successful_records": [
                    {"item_id": "smiles_0000", "sdf_artifact": None},
                    {"item_id": "smiles_0001", "sdf_artifact": None},
                ]
            },
            "artifact_files": [
                {
                    "path": "result.sdf",
                    "semantic_type": "SDFFile",
                    "media_type": "chemical/x-mdl-sd",
                },
                {
                    "path": "record-0.sdf",
                    "semantic_type": "SDFRecordFile",
                    "media_type": "chemical/x-mdl-sd",
                    "result_pointer": "/successful_records/0/sdf_artifact",
                },
                {
                    "path": "record-1.sdf",
                    "semantic_type": "SDFRecordFile",
                    "media_type": "chemical/x-mdl-sd",
                    "result_pointer": "/successful_records/0/sdf_artifact",
                },
            ],
        },
    )

    assert response["status"] == "failed"
    assert response["error"]["code"] == "result_pointer_duplicate"
    assert response["error"]["duplicate_of_artifact_index"] == 1


def test_service_rejects_result_pointer_to_missing_result_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """结果指针不能隐式创建不存在的字段。"""

    (tmp_path / "result.sdf").write_text("primary\n", encoding="utf-8")
    (tmp_path / "record-0.sdf").write_text("record\n", encoding="utf-8")
    response = _stub_service_dispatch(
        monkeypatch,
        tmp_path,
        {
            "status": "success",
            "result": {
                "successful_records": [
                    {"item_id": "smiles_0000", "sdf_artifact": None}
                ]
            },
            "artifact_files": [
                {
                    "path": "result.sdf",
                    "semantic_type": "SDFFile",
                    "media_type": "chemical/x-mdl-sd",
                },
                {
                    "path": "record-0.sdf",
                    "semantic_type": "SDFRecordFile",
                    "media_type": "chemical/x-mdl-sd",
                    "result_pointer": "/successful_records/0/not_declared",
                },
            ],
        },
    )

    assert response["status"] == "failed"
    assert response["error"]["code"] == "result_pointer_target_missing"


def test_service_without_result_pointer_preserves_existing_result_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """没有声明 result_pointer 的后端结果保持原有内容和产物顺序。"""

    (tmp_path / "result.sdf").write_text("primary\n", encoding="utf-8")
    response = _stub_service_dispatch(
        monkeypatch,
        tmp_path,
        {
            "status": "success",
            "result": {"input_count": 1, "molecules_written": 1},
            "artifact_files": [
                {
                    "path": "result.sdf",
                    "semantic_type": "SDFFile",
                    "media_type": "chemical/x-mdl-sd",
                }
            ],
        },
    )

    assert response["status"] == "success"
    assert response["result"] == {"input_count": 1, "molecules_written": 1}
    assert response["output_artifacts"][0]["semantic_type"] == "SDFFile"
