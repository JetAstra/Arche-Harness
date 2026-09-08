"""声明格式转换工具对应的 MiniChem Action。"""

from __future__ import annotations

from .._spec_builders import action as _action


ACTION_SPECS = (
    _action(
        "smiles2sdf",
        "scientific_data_interchange",
        (
            "将一个或多个 SMILES 转换为带三维坐标的 SDF；保留成功记录的聚合文件，"
            "并为每个成功项目返回可逐项交接的单记录 SDF，同时报告解析或嵌入失败。"
        ),
        "SDFFile",
        ("format_conversion",),
        ("smiles_list",),
        (),
        input_description=(
            "smiles_list: 转换工具接受的非空 SMILES 字符串列表；"
            "MCP 在工作区内分配 SDF 输出路径。"
        ),
        selection_policy="internal_deterministic",
        batch_safe=True,
    ),
    _action(
        "sdf_to_gjf",
        "scientific_data_interchange",
        (
            "读取工作区内一个带坐标的单记录 SDF，并按显式 route 参数生成 Gaussian "
            "输入文件；只生成文件，不运行 Gaussian。"
        ),
        "GaussianInputDeck",
        ("format_conversion",),
        ("sdf_path",),
        ("route_parameters", "title"),
        input_description=(
            "sdf_path: 工作区相对 SDF 路径或 SDF ArtifactRef；route_parameters 提供显式 "
            "Gaussian route 参数。"
        ),
        selection_policy="internal_deterministic",
    ),
    _action(
        "xyz_to_gjf",
        "scientific_data_interchange",
        "读取工作区内一个 XYZ 坐标文件，并按显式 route、charge 和 multiplicity 生成 Gaussian 输入文件；只生成文件，不运行 Gaussian。",
        "GaussianInputDeck",
        ("format_conversion",),
        ("input_xyz_path",),
        (
            "route_section",
            "charge",
            "multiplicity",
            "title",
            "chk_file",
            "fix_atom_count_line",
            "write_back_fixed_xyz",
        ),
        input_description=(
            "input_xyz_path: 工作区相对 XYZ 路径或 XYZFile/AtomicStructureFile ArtifactRef；route_section 可直接绑定 "
            "显式 Gaussian route 内容。"
        ),
        selection_policy="internal_deterministic",
    ),
)


__all__ = ["ACTION_SPECS"]
