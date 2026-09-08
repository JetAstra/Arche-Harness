"""三维结构生成必须保留调用方声明的电子状态。"""

from __future__ import annotations

from pathlib import Path

from minichem_toolbox.backends import structure


def _request(molecule: dict[str, object], *, force_field: str | None = None) -> dict:
    """构造不依赖服务层的结构生成请求。"""

    return {
        "inputs": {"molecule": molecule},
        "method_spec": {"force_field": force_field} if force_field else {},
        "action_settings": {"random_seed": 19},
        "resource_limits": {"walltime_seconds": 30},
    }


def test_rdkit_3d_generation_preserves_explicit_charge_and_multiplicity(
    monkeypatch, tmp_path
) -> None:
    """RDKit 生成坐标后不得把调用方的中性双重态改成单重态。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = structure.execute(
        "generate_3d_structure",
        "rdkit",
        _request({"smiles": "C", "charge": 0, "multiplicity": 2}),
    )

    assert response["status"] == "success"
    generated = response["result"]["structure"]
    assert generated["charge"] == 0
    assert generated["multiplicity"] == 2


def test_openbabel_3d_generation_preserves_state_after_sdf_parsing(
    monkeypatch, tmp_path
) -> None:
    """OpenBabel 输出经 SDF 解析后仍保留输入 mapping 的电子状态。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    def fake_run_external(*, arguments, **_kwargs):
        output_path = Path(arguments[arguments.index("-O") + 1])
        output_path.write_text(
            "\n".join(
                [
                    "generated",
                    "  MiniChem",
                    "",
                    "  1  0  0  0  0  0            999 V2000",
                    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                    "M  END",
                    "$$$$",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "available": True,
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "command": ["obabel", *arguments],
        }

    monkeypatch.setattr(structure, "run_external", fake_run_external)

    response = structure.execute(
        "generate_3d_structure",
        "openbabel",
        _request(
            {"smiles": "C", "charge": 0, "multiplicity": 2},
            force_field="uff",
        ),
    )

    assert response["status"] == "success"
    generated = response["result"]["structure"]
    assert generated["charge"] == 0
    assert generated["multiplicity"] == 2
