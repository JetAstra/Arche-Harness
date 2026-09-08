import argparse

from rdkit import Chem


def infer_charge_and_multiplicity(mol):
    """
    Name: infer_charge_and_multiplicity
    Description: Infer formal charge and spin multiplicity from an RDKit molecule.
    Parameters:
    mol: Chem.Mol RDKit molecule object.
    Returns:
    tuple Inferred (charge, multiplicity).
    """
    charge = Chem.GetFormalCharge(mol)
    num_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    multiplicity = num_radicals + 1
    return charge, multiplicity


def sdf_to_gjf(
    sdf_path,
    gjf_path,
    route_parameters="p UB3LYP/6-31+G(d,p) empiricaldispersion=gd3bj opt=(calcfc) freq stable=opt integral=ultrafine scrf=(smd,solvent=ethylacetate)",
    title="Generated from SDF",
    nprocshared=8,
    mem="4GB",
):
    """
    Name: sdf_to_gjf
    Description: Convert a single-conformer SDF file into a Gaussian .gjf file.
    Parameters:
    sdf_path: str Input SDF file path.
    gjf_path: str Output Gaussian .gjf path.
    route_parameters: str Gaussian route section content with or without leading #.
    title: str Title line of Gaussian input.
    nprocshared: int Number of shared processors for Gaussian.
    mem: str Gaussian memory specification.
    Returns:
    str Path to generated .gjf file.
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = suppl[0] if len(suppl) > 0 else None
    if mol is None:
        raise ValueError(f"Failed to read SDF: {sdf_path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"SDF has no conformer coordinates: {sdf_path}")

    conf = mol.GetConformer()
    charge, multiplicity = infer_charge_and_multiplicity(mol)

    normalized_route = route_parameters.strip()
    if normalized_route.startswith("#"):
        route_line = normalized_route
    else:
        route_line = f"# {normalized_route}"

    lines = [
        f"%nprocshared={nprocshared}",
        f"%mem={mem}",
        route_line,
        "",
        title,
        "",
        f"{charge} {multiplicity}",
    ]

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        lines.append(f"{symbol:2}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}")

    lines.append("")

    with open(gjf_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n\n")

    return gjf_path


def _build_cli_parser():
    parser = argparse.ArgumentParser(description="Convert SDF to Gaussian .gjf")
    parser.add_argument("sdf_path", help="Input SDF file path")
    parser.add_argument("gjf_path", help="Output .gjf path")
    parser.add_argument(
        "--route_parameters",
        default="p UB3LYP/6-31+G(d,p) empiricaldispersion=gd3bj opt=(calcfc) freq stable=opt integral=ultrafine scrf=(smd,solvent=ethylacetate)",
    )
    parser.add_argument("--title", default="Generated from SDF")
    parser.add_argument("--nprocshared", type=int, default=8)
    parser.add_argument("--mem", default="4GB")
    return parser


if __name__ == "__main__":
    args = _build_cli_parser().parse_args()
    output = sdf_to_gjf(
        sdf_path=args.sdf_path,
        gjf_path=args.gjf_path,
        route_parameters=args.route_parameters,
        title=args.title,
        nprocshared=args.nprocshared,
        mem=args.mem,
    )
    print(output)
