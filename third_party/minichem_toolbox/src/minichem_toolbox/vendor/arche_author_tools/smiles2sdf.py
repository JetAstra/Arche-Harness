import argparse
from typing import Iterable, List

from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter


def smiles_to_sdf(smiles_list: Iterable[str], output_sdf_path: str) -> str:
    """
    Name: smiles_to_sdf
    Description: Convert one or more SMILES strings into a 3D SDF file while preserving radical electron annotations.
    Parameters:
    smiles_list: Iterable[str] SMILES strings to convert.
    output_sdf_path: str Output SDF path.
    Returns:
    str Output SDF path.
    """
    writer = SDWriter(output_sdf_path)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Failed to parse SMILES: {smi}")
            continue

        radical_info = {atom.GetIdx(): atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()}

        mol = Chem.AddHs(mol)
        for idx, num_electrons in radical_info.items():
            if num_electrons > 0:
                mol.GetAtomWithIdx(idx).SetNumRadicalElectrons(num_electrons)

        embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if embed_status != 0:
            print(f"3D embedding failed for SMILES: {smi}")
            continue

        writer.write(mol)

    writer.close()
    return output_sdf_path


def _parse_smiles_argument(smiles_args: List[str]) -> List[str]:
    # Allow either repeated --smiles values or comma-separated values.
    result: List[str] = []
    for item in smiles_args:
        for token in item.split(","):
            token = token.strip()
            if token:
                result.append(token)
    return result


def _build_cli_parser():
    parser = argparse.ArgumentParser(description="Convert SMILES to SDF")
    parser.add_argument(
        "--smiles",
        nargs="+",
        required=True,
        help="SMILES inputs (repeat or comma-separate)",
    )
    parser.add_argument("--output", required=True, help="Output SDF path")
    return parser


if __name__ == "__main__":
    cli_args = _build_cli_parser().parse_args()
    smiles_values = _parse_smiles_argument(cli_args.smiles)
    output = smiles_to_sdf(smiles_values, cli_args.output)
    print(output)
