import yaml
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBParser
from Bio.Data.IUPACData import protein_letters_3to1
import numpy as np
from Bio import SeqIO, AlignIO


def write_clustered_fasta(clusters, save_path: Path, ligand_sequence: str | None = None):
    """
    Generate fasta files with optional second protein (ligand).
    """

    # --- adjust filename ---
    if ligand_sequence:
        save_path = save_path.with_name(save_path.stem + "_with_ligand.fasta")

    with save_path.open("w") as fh:
        for cluster_label in sorted(clusters):
            seq_ids = clusters[cluster_label]["ids"]
            sequences = clusters[cluster_label]["sequences"]

            for i, (orig_id, seq) in enumerate(zip(seq_ids, sequences), start=1):
                seq_clean = seq.replace("-", "")

                header = f">cluster_{cluster_label}_seq{i}|orig_id={orig_id}"

                if ligand_sequence:
                    header += "|with_ligand"

                fh.write(header + "\n")

                if ligand_sequence:
                    fh.write(seq_clean + ":" + ligand_sequence + "\n")
                else:
                    fh.write(seq_clean + "\n")

    print(f"Saved clustered fasta file for esmFold to {save_path}")


def write_clustered_yaml(clusters, save_dir: Path, ligand_sequence: str | None = None):
    """
    Generate one YAML file per sequence for Boltz.

    Supports optional second protein.
    """

    # --- adjust directory name ---
    if ligand_sequence:
        save_dir = save_dir.parent / (save_dir.name + "_with_ligand")

    save_dir.mkdir(parents=True, exist_ok=True)

    for cluster_label in sorted(clusters):
        seq_ids = clusters[cluster_label]["ids"]
        sequences = clusters[cluster_label]["sequences"]

        for orig_id, seq in zip(seq_ids, sequences):
            seq_clean = seq.replace("-", "")

            filename = f"cluster{cluster_label}_{orig_id}"
            if ligand_sequence:
                filename += "_with_ligand"
            filename += ".yaml"

            yaml_path = save_dir / filename

            if ligand_sequence:
                data = {
                    "sequences": [
                        {
                            "protein": {
                                "id": "A",
                                "sequence": seq_clean,
                                "cyclic": False,
                                "msa": "empty"
                            }
                        },
                        {
                            "protein": {
                                "id": "B",
                                "sequence": ligand_sequence,
                                "cyclic": False,
                                "msa": "empty"
                            }
                        }
                    ]
                }
            else:
                data = {
                    "sequences": [
                        {
                            "protein": {
                                "id": "A",
                                "sequence": seq_clean,
                                "cyclic": False,
                                "msa": "empty"
                            }
                        }
                    ]
                }

            with yaml_path.open("w") as fh:
                yaml.dump(data, fh, sort_keys=False)

    print(f"Saved individual YAML files to {save_dir}")


def find_disulfide_bonds(structure_path: Path, cutoff: float = 2.1):
    """
    Identify disulfide bonds based on SG–SG distance.

    Parameters
    ----------
    structure_path : Path
        Path to .cif or .pdb file
    cutoff : float
        Distance cutoff in Å (default: 2.1 Å)

    Returns
    -------
    list of tuples:
        [(res1_id, res2_id, distance)]
    """

    # --- choose parser automatically ---
    if structure_path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    elif structure_path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format")

    structure = parser.get_structure("struct", str(structure_path))

    cysteines = []

    # --- collect all cysteine SG atoms ---
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "CYS" and "SG" in residue:
                    cysteines.append((chain.id, residue.get_id()[1], residue["SG"]))

    disulfides = []

    # --- compute pairwise distances ---
    for i in range(len(cysteines)):
        for j in range(i + 1, len(cysteines)):

            chain1, res1_id, atom1 = cysteines[i]
            chain2, res2_id, atom2 = cysteines[j]

            dist = np.linalg.norm(atom1.coord - atom2.coord)

            if dist < cutoff:
                disulfides.append(
                    (chain1, res1_id, chain2, res2_id, dist)
                )

    return disulfides


def collect_disulfides_from_folder(pdb_folder: Path, cutoff: float = 2.1, debug: bool = False):
    """
    Apply disulfide detection to all structures in a folder.

    Parameters
    ----------
    pdb_folder : Path
        Folder containing .pdb or .cif files
    cutoff : float
        Distance cutoff for disulfide bonds (default: 2.1 Å)

    Returns
    -------
    dict:
        {
            "file1.cif": [(chain1, res1, chain2, res2, dist), ...],
            "file2.pdb": [...],
        }
    """

    results = {}

    for file in pdb_folder.iterdir():

        if file.suffix.lower() not in [".pdb", ".cif"]:
            continue

        bonds = find_disulfide_bonds(file, cutoff=cutoff)

        if bonds:
            results[file.name] = bonds

            if debug:
                print(f"\n[FILE] {file.name}")
                for b in bonds:
                    print(
                        f"  CYS {b[1]} ({b[0]}) ↔ CYS {b[3]} ({b[2]}) "
                        f"dist = {b[4]:.2f} Å")
        else:
            if debug:
                print(f"\n[FILE] {file.name} → no disulfide bonds found")

    return results


def extract_sequence_from_structure(structure_path: Path):
    """
    Extract sequence AND mapping from PDB residue numbering
    to continuous sequence index.

    Returns:
        seq (str)
        residue_map (dict): {pdb_residue_id → sequence_index}
    """

    if structure_path.suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("x", str(structure_path))

    seq = []
    residue_map = {}

    counter = 0

    for model in structure:
        for chain in model:
            for res in chain:

                if res.get_resname() == "HOH": # ignores water
                    continue

                aa = protein_letters_3to1.get(res.get_resname().capitalize(), "") # three to one letter code
                if not aa:
                    continue

                counter += 1

                pdb_id = res.get_id()[1]   # e.g. 103
                residue_map[pdb_id] = counter

                seq.append(aa)

    return "".join(seq), residue_map


def map_disulfides_to_knobs(pdb_folder: Path, fasta_path: Path, disulfides: dict, debug: bool = False,
):
    """
    Map precomputed disulfides to knob IDs using sequence matching
    and convert PDB residue numbering → sequence index.
    """

    # --- seq from FASTA ---
    seq_to_knob = {
        str(record.seq): record.id
        for record in SeqIO.parse(fasta_path, "fasta")
    }

    if debug:
        print("\n=== Mapping disulfides to knobs START ===")
        print(f"Loaded {len(seq_to_knob)} knob sequences from FASTA")

    results = {}

    for file_name, bonds in disulfides.items():

        file_path = pdb_folder / file_name

        # --- extract sequence and mapping ---
        seq, residue_map = extract_sequence_from_structure(file_path)

        if debug:
            print(f"\n[STRUCTURE] {file_name}")
            print(f"Extracted sequence length: {len(seq)}")
            print(f"Residue map size: {len(residue_map)}")

        if seq not in seq_to_knob:
            print(f"[WARN] No match for {file_name}")
            continue

        knob_id = seq_to_knob[seq]

        # --- remap disulfides ---
        remapped_bonds = []

        for chain1, res1, chain2, res2, dist in bonds:

            mapped_res1 = residue_map.get(res1)
            mapped_res2 = residue_map.get(res2)

            if debug:
                print(
                    f"  [MAP] PDB ({res1}, {res2}) → "
                    f"SEQ ({mapped_res1}, {mapped_res2})")

            # --- check mapping ---
            if mapped_res1 is None or mapped_res2 is None:
                if debug:
                    print("    -> SKIP: residue not found in mapping")
                continue

            if mapped_res1 > len(seq) or mapped_res2 > len(seq):
                if debug:
                    print("    -> SKIP: mapped position out of bounds")
                continue

            remapped_bonds.append(
                (chain1, mapped_res1, chain2, mapped_res2, dist))

        results[knob_id] = {
            "sequence": seq,
            "structure_file": file_name,
            "disulfides": remapped_bonds}   # mapped positions!

        # --- debug summary per structure ---
        if debug:
            print(f"\n[{knob_id}] ← {file_name}")
            print(f"Sequence: {seq}")

            if remapped_bonds:
                for b in remapped_bonds:
                    print(
                        f"  CYS {b[1]} ↔ CYS {b[3]} "
                        f"({b[4]:.2f} Å)")
            else:
                print("  No valid disulfides after mapping")

    if debug:
        print("\n=== Mapping disulfides to knobs END ===")

    return results


def generate_valid_disulfide_constraints(near_knobs_to_knobs: dict, disulfides_mapped_knobs: dict, clusters: dict,
                                         alignment_path: Path, min_seq_sep: int = 3, debug: bool = False
):
    """
    Generate valid disulfide constraints for near-knobs by transferring
    disulfides from reference knobs using alignment-based mapping.
    """

    # --- build sequence dict from clusters: extract sequences ---
    sequences = {}
    for cluster_data in clusters.values():
        for i, seq_id in enumerate(cluster_data["ids"]):
            if seq_id in sequences:
                print(f"[WARN] duplicate ID: {seq_id}")
            sequences[seq_id] = cluster_data["sequences"][i]

    # --- load alignment ---
    alignment = AlignIO.read(alignment_path, "clustal")
    aln_dict = {rec.id: str(rec.seq) for rec in alignment}

    # --- helper: gives position of residue in ref_aln which corresponds to ref_pos (C-pos)---
    def map_position(ref_aln, target_aln, ref_pos):
        count_ref = 0
        count_target = 0
        for i in range(len(ref_aln)):
            if ref_aln[i] != "-": # counts aa in ref_aln
                count_ref += 1
            if target_aln[i] != "-": # counts aa in target_aln
                count_target += 1
            if count_ref == ref_pos: # when requested C-position is reached, return corresponding  position in target_aln
                return count_target if target_aln[i] != "-" else None
        return None

    # --- helper: ---
    def aln_to_seq_index(aln_seq, aln_pos):
        count = 0
        for k in range(aln_pos + 1):
            if aln_seq[k] != "-":
                count += 1
        return count

    results = {}

    for near_id, knob_list in near_knobs_to_knobs.items(): # for every near_knob

        if near_id not in sequences or near_id not in aln_dict:
            continue

        near_aln = aln_dict[near_id]

        valid_constraints = set()
        used_positions = set()

        if debug:
            print(f"\n=== Near-knob: {near_id} ===")

        for knob_id in knob_list: # for every knob near to this near_knob

            if knob_id not in disulfides_mapped_knobs:
                continue
            if knob_id not in aln_dict:
                continue

            knob_aln = aln_dict[knob_id]
            ds_list = disulfides_mapped_knobs[knob_id]["disulfides"]

            for (_, i, _, j, _) in ds_list: # chain1, !res1_id!, chain2, !res2_id!, dist

                # gives position of residue in knob_aln which corresponds to C-pos of knob
                mapped_i = map_position(knob_aln, near_aln, i)
                mapped_j = map_position(knob_aln, near_aln, j)

                if debug:
                    print(f"\n[REF] {knob_id}")
                    print(f"Original DS: ({i}, {j})")
                    print(f"Mapped → ({mapped_i}, {mapped_j})")

                # --- CHECK 1: mapping exists (no gap)---
                if mapped_i is None or mapped_j is None:
                    if debug:
                        print("  -> REJECTED: gap in near-knob")
                    continue

                # --- CHECK 2: within sequence (not too big) ---
                if mapped_i > len(near_aln) or mapped_j > len(near_aln):
                    if debug:
                        print("  -> REJECTED: out of sequence")
                    continue

                # --- CHECK 3: are both cysteines present ---
                if near_aln[mapped_i - 1] != "C" or near_aln[mapped_j - 1] != "C":
                    if debug:
                        print("  -> REJECTED: no C in near-knob")
                    continue

                # --- CHECK 4: above minimal seq distance ---
                seq_i = aln_to_seq_index(near_aln, mapped_i - 1)
                seq_j = aln_to_seq_index(near_aln, mapped_j - 1)

                if abs(seq_i - seq_j) < min_seq_sep:
                    if debug:
                        print("  -> REJECTED: too close in sequence")
                    continue

                # --- CHECK 5: no reuse of cysteine ---
                if mapped_i in used_positions or mapped_j in used_positions:
                    if debug:
                        print("  -> REJECTED: cysteine reused")
                    continue

                valid_constraints.add((mapped_i, mapped_j)) # stores both connected near knob Cs together
                used_positions.update([mapped_i, mapped_j])
                if debug:
                    print(f"  -> ACCEPTED: Cys {mapped_i} ↔ Cys {mapped_j}")

        results[near_id] = sorted(valid_constraints)
        if debug:
            print(f"\nFinal constraints: {sorted(valid_constraints)}")

    return results
