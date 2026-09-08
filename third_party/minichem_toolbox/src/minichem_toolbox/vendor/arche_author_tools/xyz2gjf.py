import argparse
import json
import os
from io import StringIO
from typing import Dict, List, Optional, Sequence, Tuple

from ase.io import read


DEFAULT_ROUTE_SECTION = "#p opt freq B3LYP/6-31G(d) scrf=(iefpcm,solvent=acetone) Temperature=303.15"
DEFAULT_TARGET_FILENAMES = ("product.xyz", "reactant.xyz")


def _normalize_route_section(route_section: str) -> str:
    route_text = route_section.strip()
    if not route_text.startswith("#"):
        route_text = f"# {route_text}"
    return route_text


def _read_xyz_with_optional_fix(
    input_xyz_path: str,
    fix_atom_count_line: bool = True,
    write_back_fixed_xyz: bool = False,
) -> Tuple[object, bool, Optional[int]]:
    with open(input_xyz_path, "r", encoding="utf-8") as file_obj:
        lines = file_obj.readlines()

    atom_count_fixed = False
    fixed_atom_count = None

    if fix_atom_count_line and lines:
        first_line = lines[0].strip()
        if "." in first_line:
            try:
                fixed_atom_count = int(float(first_line))
                lines[0] = f"{fixed_atom_count}\n"
                atom_count_fixed = True
            except ValueError:
                atom_count_fixed = False

    xyz_text = "".join(lines)
    atoms = read(StringIO(xyz_text), format="xyz")

    if atom_count_fixed and write_back_fixed_xyz:
        with open(input_xyz_path, "w", encoding="utf-8") as file_obj:
            file_obj.writelines(lines)

    return atoms, atom_count_fixed, fixed_atom_count


def xyz_to_gjf(
    input_xyz_path: str,
    output_gjf_path: str,
    route_section: str = DEFAULT_ROUTE_SECTION,
    charge: int = 0,
    multiplicity: int = 1,
    title: str = "Converted from XYZ file",
    chk_file: Optional[str] = None,
    nprocshared: Optional[int] = None,
    mem: Optional[str] = None,
    fix_atom_count_line: bool = True,
    write_back_fixed_xyz: bool = False,
) -> Dict:
    """
    Name: xyz_to_gjf
    Description: Convert a single XYZ file to one Gaussian GJF file.
    Parameters:
    input_xyz_path: str Input XYZ file path.
    output_gjf_path: str Output GJF file path.
    route_section: str Gaussian route section.
    charge: int Molecular charge.
    multiplicity: int Spin multiplicity.
    title: str Gaussian title line.
    chk_file: Optional[str] Checkpoint filename. If None, auto-generated from output filename.
    nprocshared: Optional[int] Gaussian %nprocshared value.
    mem: Optional[str] Gaussian %mem value.
    fix_atom_count_line: bool Whether to fix a decimal atom-count line in XYZ input.
    write_back_fixed_xyz: bool Whether to write the fixed XYZ atom-count line back to input file.
    Returns:
    dict Structured conversion result.
    """
    result = {
        "success": False,
        "input_xyz_path": input_xyz_path,
        "output_gjf_path": output_gjf_path,
        "route_section": _normalize_route_section(route_section),
        "charge": charge,
        "multiplicity": multiplicity,
        "atom_count_line_fixed": False,
        "write_back_fixed_xyz": write_back_fixed_xyz,
        "atoms_written": 0,
        "message": "",
    }

    try:
        atoms, atom_count_fixed, fixed_atom_count = _read_xyz_with_optional_fix(
            input_xyz_path=input_xyz_path,
            fix_atom_count_line=fix_atom_count_line,
            write_back_fixed_xyz=write_back_fixed_xyz,
        )
        result["atom_count_line_fixed"] = atom_count_fixed
        if fixed_atom_count is not None:
            result["fixed_atom_count"] = fixed_atom_count

        os.makedirs(os.path.dirname(output_gjf_path) or ".", exist_ok=True)

        chk_value = chk_file
        if chk_value is None:
            chk_value = f"{os.path.splitext(os.path.basename(output_gjf_path))[0]}.chk"

        lines: List[str] = []
        if chk_value:
            lines.append(f"%chk={chk_value}")
        if nprocshared is not None:
            lines.append(f"%nprocshared={nprocshared}")
        if mem is not None:
            lines.append(f"%mem={mem}")

        lines.append(result["route_section"])
        lines.append("")
        lines.append(title)
        lines.append("")
        lines.append(f"{charge} {multiplicity}")

        for atom in atoms:
            lines.append(f"{atom.symbol}   {atom.x:.6f}   {atom.y:.6f}   {atom.z:.6f}")

        lines.append("")

        with open(output_gjf_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(lines))

        result["atoms_written"] = len(atoms)
        result["success"] = True
        result["message"] = "Converted XYZ to GJF successfully"
        return result
    except Exception as exc:
        result["message"] = str(exc)
        return result


def batch_convert_xyz_to_gjf(
    root_dir: str,
    target_filenames: Sequence[str] = DEFAULT_TARGET_FILENAMES,
    route_section: str = DEFAULT_ROUTE_SECTION,
    charge: int = 0,
    multiplicity: int = 1,
    title: str = "Converted from XYZ file",
    chk_file: Optional[str] = None,
    nprocshared: Optional[int] = None,
    mem: Optional[str] = None,
    recursive: bool = True,
    fix_atom_count_line: bool = True,
    write_back_fixed_xyz: bool = False,
) -> Dict:
    """
    Name: batch_convert_xyz_to_gjf
    Description: Optionally batch-convert selected XYZ files under a directory tree.
    Parameters:
    root_dir: str Root directory to scan.
    target_filenames: Sequence[str] XYZ filenames to include.
    route_section: str Gaussian route section.
    charge: int Molecular charge.
    multiplicity: int Spin multiplicity.
    title: str Gaussian title line.
    chk_file: Optional[str] Optional fixed checkpoint filename.
    nprocshared: Optional[int] Gaussian %nprocshared value.
    mem: Optional[str] Gaussian %mem value.
    recursive: bool Whether to traverse subdirectories.
    fix_atom_count_line: bool Whether to fix decimal atom-count lines in XYZ input.
    write_back_fixed_xyz: bool Whether to write fixed atom-count lines back to XYZ files.
    Returns:
    dict Batch summary and per-file results.
    """
    batch_result: Dict = {
        "success": True,
        "root_dir": root_dir,
        "target_filenames": list(target_filenames),
        "total_candidates": 0,
        "success_count": 0,
        "failure_count": 0,
        "results": [],
        "message": "",
    }

    try:
        targets = set(target_filenames)

        if recursive:
            walker = os.walk(root_dir)
        else:
            filenames = [name for name in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, name))]
            walker = [(root_dir, [], filenames)]

        for subdir, _, files in walker:
            for filename in files:
                if filename not in targets:
                    continue

                batch_result["total_candidates"] += 1
                input_xyz_path = os.path.join(subdir, filename)
                output_gjf_path = os.path.join(subdir, f"{os.path.splitext(filename)[0]}.gjf")

                conversion_result = xyz_to_gjf(
                    input_xyz_path=input_xyz_path,
                    output_gjf_path=output_gjf_path,
                    route_section=route_section,
                    charge=charge,
                    multiplicity=multiplicity,
                    title=title,
                    chk_file=chk_file,
                    nprocshared=nprocshared,
                    mem=mem,
                    fix_atom_count_line=fix_atom_count_line,
                    write_back_fixed_xyz=write_back_fixed_xyz,
                )
                batch_result["results"].append(conversion_result)

                if conversion_result.get("success"):
                    batch_result["success_count"] += 1
                else:
                    batch_result["failure_count"] += 1

        if batch_result["failure_count"] > 0:
            batch_result["success"] = False
            batch_result["message"] = "Batch conversion finished with failures"
        else:
            batch_result["message"] = "Batch conversion finished successfully"

        return batch_result
    except Exception as exc:
        batch_result["success"] = False
        batch_result["message"] = str(exc)
        return batch_result


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert XYZ files to Gaussian GJF")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser("single", help="Convert one XYZ file")
    single_parser.add_argument("input_xyz_path", help="Input XYZ path")
    single_parser.add_argument("output_gjf_path", help="Output GJF path")

    batch_parser = subparsers.add_parser("batch", help="Batch convert XYZ files in a directory")
    batch_parser.add_argument("root_dir", help="Root directory")
    batch_parser.add_argument(
        "--target_filenames",
        nargs="+",
        default=list(DEFAULT_TARGET_FILENAMES),
        help="Target XYZ filenames to process",
    )
    batch_parser.add_argument("--non_recursive", action="store_true", help="Disable recursive traversal")

    for subparser in (single_parser, batch_parser):
        subparser.add_argument("--route_section", default=DEFAULT_ROUTE_SECTION)
        subparser.add_argument("--charge", type=int, default=0)
        subparser.add_argument("--multiplicity", type=int, default=1)
        subparser.add_argument("--title", default="Converted from XYZ file")
        subparser.add_argument("--chk_file", default=None)
        subparser.add_argument("--nprocshared", type=int, default=None)
        subparser.add_argument("--mem", default=None)
        subparser.add_argument("--no_fix_atom_count_line", action="store_true")
        subparser.add_argument("--write_back_fixed_xyz", action="store_true")

    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()

    if args.mode == "single":
        output = xyz_to_gjf(
            input_xyz_path=args.input_xyz_path,
            output_gjf_path=args.output_gjf_path,
            route_section=args.route_section,
            charge=args.charge,
            multiplicity=args.multiplicity,
            title=args.title,
            chk_file=args.chk_file,
            nprocshared=args.nprocshared,
            mem=args.mem,
            fix_atom_count_line=not args.no_fix_atom_count_line,
            write_back_fixed_xyz=args.write_back_fixed_xyz,
        )
    else:
        output = batch_convert_xyz_to_gjf(
            root_dir=args.root_dir,
            target_filenames=args.target_filenames,
            route_section=args.route_section,
            charge=args.charge,
            multiplicity=args.multiplicity,
            title=args.title,
            chk_file=args.chk_file,
            nprocshared=args.nprocshared,
            mem=args.mem,
            recursive=not args.non_recursive,
            fix_atom_count_line=not args.no_fix_atom_count_line,
            write_back_fixed_xyz=args.write_back_fixed_xyz,
        )

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
