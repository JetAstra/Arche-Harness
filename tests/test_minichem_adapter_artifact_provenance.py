from chemistry_multiagent.tools.minichem_adapter import _validate_artifact_records


def _artifact(producer_action: str, producer_backend: str, suffix: str) -> dict:
    artifact_id = "art_0123456789abcdef0123456789abcdef"[:-1] + suffix
    return {
        "artifact_id": artifact_id,
        "media_type": "application/json",
        "parent_artifact_ids": [],
        "path": f"_tool_artifacts/objects/{artifact_id}.json",
        "producer_action": producer_action,
        "producer_backend": producer_backend,
        "semantic_type": "AtomicStructure",
        "sha256": suffix * 64,
    }


def test_input_artifact_keeps_upstream_producer_identity() -> None:
    result = _validate_artifact_records(
        {
            "status": "success",
            "input_artifacts": [_artifact("standardize_structure", "rdkit", "a")],
            "output_artifacts": [_artifact("generate_3d_structure", "rdkit", "b")],
        },
        action_id="generate_3d_structure",
        backend_id="rdkit",
    )

    assert result["input_artifacts"][0]["producer_action"] == "standardize_structure"
    assert result["output_artifacts"][0]["producer_action"] == "generate_3d_structure"
