import shutil
from pathlib import Path
from datetime import datetime

from Bio.PDB import MMCIFParser, PDBIO
from structure_prediction_analysis import extract_seq_id
from pdbfixer import PDBFixer
from openmm.app import PDBFile


def archive_and_clean_md_root(md_root: Path, archive_root: Path
):
    archive_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = archive_root / timestamp

    archived = 0
    removed = 0

    for seq_dir in md_root.glob("seq_*"):

        if not seq_dir.is_dir():
            continue

        if contains_md_results(seq_dir):
            archive_dir.mkdir(exist_ok=True)
            shutil.move(
                str(seq_dir),
                str(archive_dir / seq_dir.name))
            archived += 1
            print(f"[MD] Archived: {seq_dir.name}")

        else:
            shutil.rmtree(seq_dir)
            removed += 1
            print(f"[MD] Removed unused folder: "f"{seq_dir.name}")
    print(
        f"[MD] Cleanup finished "
        f"(archived={archived}, removed={removed})")


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path
):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", str(cif_path))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))

    print(f"[MD] Wrote: {pdb_path}")


def fix_structure_for_md(input_pdb: Path, output_pdb: Path, ph: float = 7.4,
):
    """
    Make AlphaFold/Boltz structures more MD-ready.

    - find missing residues
    - find missing atoms
    - add missing atoms
    - add hydrogens
    """

    fixer = PDBFixer(filename=str(input_pdb))

    fixer.findMissingResidues()
    fixer.findMissingAtoms()

    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with open(output_pdb, "w") as handle:
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            handle
        )

    print(f"[MD] Fixed structure: {output_pdb}")


def prepare_md_input_for_seq(seq_id: str, requested_model: str, ligand: bool, model_database: dict, md_root: Path,
):

    candidates = []

    # ----------------------------
    # AF3 candidates
    # ----------------------------
    af_stats = model_database[("AF3", ligand)]

    for mid, data in af_stats.items():

        try:
            current_seq = extract_seq_id(mid)
        except ValueError:
            continue

        if current_seq != seq_id:
            continue

        candidates.append({
            "model": "AF3",
            "score": data["mean"],
            "path": data["cif_path"],})

    # ----------------------------
    # Boltz candidates
    # ----------------------------
    boltz_stats = model_database[("Boltz", ligand)]

    for mid, data in boltz_stats.items():

        try:
            current_seq = extract_seq_id(mid)
        except ValueError:
            continue

        if current_seq != seq_id:
            continue

        candidates.append({
            "model": "Boltz",
            "score": data["mean"],
            "path": data["cif_path"],})

    # ----------------------------
    # no structure found
    # ----------------------------
    if not candidates:
        raise ValueError(f"No structure found for {seq_id}")

    # ----------------------------
    # select requested model
    # ----------------------------
    selected = None

    if requested_model.upper() in ["AF3", "BOLTZ"]:
        matching = [
            c for c in candidates
            if c["model"].upper() == requested_model.upper()]

        if matching:
            selected = max(matching,key=lambda x: x["score"])

        else:
            print(
                f"[MD] {seq_id}: requested "
                f"{requested_model} not found -> "
                f"falling back to best available model")

    # ----------------------------
    # best model fallback
    # ----------------------------
    if selected is None:

        selected = max(
            candidates,
            key=lambda x: x["score"])

    # ----------------------------
    # folder naming
    # ----------------------------
    if requested_model.lower() == "best":
        selected_label = (
            "best_af"
            if selected["model"] == "AF3"
            else "best_boltz")
    else:
        selected_label = selected["model"]

    ligand_label = (
        "with_ligand"
        if ligand
        else "without_ligand")

    folder_name = (f"{seq_id}_{selected_label}_{ligand_label}")
    input_dir = md_root / folder_name

    # remove old directory completely
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)

    # ----------------------------
    # write PDB
    # ----------------------------
    raw_pdb = input_dir / "raw_input.pdb"

    convert_cif_to_pdb(selected["path"], raw_pdb)

    final_pdb = input_dir / "input.pdb"
    fix_structure_for_md(input_pdb=raw_pdb, output_pdb=final_pdb,)

    print(
        f"[MD] {seq_id}: selected "
        f"{selected['model']} "
        f"(mean pLDDT={selected['score']:.2f})")

    print(f"[MD] Output directory: {input_dir}")


def distribute_md_scripts(md_root: Path, template_dir: Path
):
    """
    Copy MD scripts (run_md.py and run_md.slurm) into each seq folder.

    Parameters:
        md_root      folder containing seq_x directories
        template_dir folder containing run_md.py and run_md.slurm
    """

    py_file = template_dir / "run_md.py"
    slurm_file = template_dir / "run_md.slurm"

    if not py_file.exists() or not slurm_file.exists():
        raise FileNotFoundError("Template run_md.py or run_md.slurm not found")

    # iterate over seq folders
    for seq_dir in md_root.glob("seq_*"):

        if not seq_dir.is_dir():
            continue

        # destination paths
        dest_py = seq_dir / "run_md.py"
        dest_slurm = seq_dir / "run_md.slurm"

        shutil.copy(py_file, dest_py)
        shutil.copy(slurm_file, dest_slurm)

        print(f"[MD] Scripts copied to: {seq_dir}")

    # check if no folders found
    if not any(md_root.glob("seq_*")):
        print("[MD] WARNING: No seq folders found!")


def contains_md_results(seq_dir: Path
) -> bool:

    md_output_patterns = [
        "*.dcd",
        "*.xtc",
        "*.nc",
        "*.out",
        "*.err",
        "log.txt",
        "*.csv",
    ]

    for pattern in md_output_patterns:

        if any(seq_dir.glob(pattern)):
            return True

    return False