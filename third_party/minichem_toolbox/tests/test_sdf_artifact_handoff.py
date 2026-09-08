"""验证 SDF ArtifactRef 能在结构生成和 xTB 优化之间安全交接。"""

from __future__ import annotations

from pathlib import Path

import pytest

from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox.backends import common, electronic, structure


_CASE3_NICKEL_COMPLEX = "Br[Ni]1([H])N2=CC=CC=C2C3=CC=CC=N31"


def _generate_case3_v3000_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """在临时任务工作区生成并登记 Case 3 镍配合物的真实 V3000 ArtifactRef。

    输入是 pytest 环境变量控制器和临时工作区；函数调用确定性的 RDKit 结构生成
    并登记输出文件，副作用仅限于临时目录。返回完整 ArtifactRef；生成结果缺失、
    不是 V3000 或登记失败时由断言或底层异常终止测试。
    """

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    generated = structure.execute(
        "generate_3d_structure",
        "rdkit",
        {
            "inputs": {"molecule": _CASE3_NICKEL_COMPLEX},
            "method_spec": {},
            "action_settings": {"random_seed": 20260822},
        },
    )
    assert generated["status"] == "success"
    assert generated["artifact_files"]
    relative_path = generated["artifact_files"][0]["path"]
    sdf_path = tmp_path / relative_path
    assert "V3000" in sdf_path.read_text(encoding="utf-8")

    reference = ArtifactStore().register_file(
        relative_path,
        semantic_type="AtomicStructure",
        producer_action="generate_3d_structure",
        producer_backend="rdkit",
        media_type="chemical/x-mdl-sdfile",
    )
    return reference


def _fake_xtb_run_external(*, arguments, directory, **_kwargs):
    """在临时目录写出模拟的 xTB 终态，保留输入 XYZ 的原子记录。

    输入是 xTB 参数和后端临时目录；函数只写测试目录中的 ``xtbopt.xyz``，返回
    固定成功响应，不启动外部程序。返回值符合 xTB 后端的最小 worker 合同；参数
    缺失或临时目录不可写时抛出文件系统异常。
    """

    input_path = Path(arguments[0])
    (directory / "xtbopt.xyz").write_text(
        input_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return {
        "available": True,
        "stdout": " TOTAL ENERGY       -1.234567890\n",
        "stderr": "",
        "returncode": 0,
        "command": ["xtb", *[str(item) for item in arguments]],
    }


def _xtb_request(reference_value):
    """构造把给定 ArtifactRef 交给 xTB 的显式优化请求。

    输入是完整、compact 或 exact artifact-id 之一；函数只在内存中组装请求，不写
    文件。返回包含结构输入、xTB 方法和优化级别的映射；本辅助函数本身不执行
    校验，非法引用由下游 ArtifactStore 或后端抛出异常。
    """

    return {
        "inputs": {"structure": reference_value},
        "method_spec": {
            "method": "gfn2-xtb",
            "charge": 0,
            "unpaired_electrons": 0,
        },
        "action_settings": {"optimization_level": "normal"},
    }


def test_case3_v3000_artifact_ref_reaches_xtb_with_full_atomic_structure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """真实 V3000 镍配合物的完整、compact、exact 引用都可交给 xTB。"""

    reference = _generate_case3_v3000_artifact(monkeypatch, tmp_path)
    monkeypatch.setattr(electronic, "run_external", _fake_xtb_run_external)

    for reference_value in (
        reference.model_dump(mode="json"),
        {"artifact_id": reference.artifact_id},
        reference.artifact_id,
    ):
        parsed = common.structure_dict(reference_value)
        assert len(parsed["atoms"]) == 23
        assert len(parsed["bonds"]) == 25
        assert {atom["element"] for atom in parsed["atoms"]} >= {"Br", "Ni", "N"}
        assert sum(bond.get("bond_type") == "dative" for bond in parsed["bonds"]) == 2
        assert any(bond["bond_order"] == 2.0 for bond in parsed["bonds"])

        result = electronic.execute(
            "optimize_geometry", "xtb", _xtb_request(reference_value)
        )
        assert result["status"] == "success"
        assert len(result["result"]["structure"]["atoms"]) == 23


def test_sdf_artifact_handoff_keeps_hash_and_workspace_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SDF 解析修复不能绕过完整引用的哈希和工作区边界核验。"""

    reference = _generate_case3_v3000_artifact(monkeypatch, tmp_path)
    tampered_hash = reference.model_copy(update={"sha256": "0" * 64}).model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        common.structure_dict(tampered_hash)

    escaped_path = reference.model_copy(update={"path": "../outside.sdf"}).model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="escapes chemistry workspace"):
        common.structure_dict(escaped_path)


def test_v2000_sdf_parser_preserves_coordinates_bonds_and_charge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """旧式 V2000 SDF 仍解析为带键和形式电荷的完整 AtomicStructure。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    path = tmp_path / "charged.sdf"
    path.write_text(
        "\n".join(
            [
                "charged molecule",
                "  MiniChem",
                "",
                "  2  1  0  0  0  0  0  0  0  0999 V2000",
                "    0.0000    0.0000    0.0000 N   0  3  0  0  0  0  0  0  0  0  0  0",
                "    1.2000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0",
                "  1  2  1  0  0  0  0",
                "M  END",
                "$$$$",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parsed = common.structure_dict("charged.sdf")

    assert parsed["atoms"] == [
        {
            "element": "N",
            "position_angstrom": [0.0, 0.0, 0.0],
            "formal_charge": 1,
        },
        {"element": "Cl", "position_angstrom": [1.2, 0.0, 0.0]},
    ]
    assert parsed["bonds"] == [
        {"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0}
    ]
    assert parsed["charge"] == 1
