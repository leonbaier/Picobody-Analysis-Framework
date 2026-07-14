from pathlib import Path
import pandas as pd
from typing import Dict, List
from Bio import SeqIO
import pickle


def save_negative_binders(negative_binder_sequences: Dict[str, Dict], save_dir: Path
) -> None:
    """
    Save experimentally confirmed negative binders as CSV.
    """

    df = pd.DataFrame([
        {
            "sequence": seq,
            "name": info["name"],
            "first_found_in": info.get("first_found_in")
        }
        for seq, info in negative_binder_sequences.items()
    ])

    df.to_csv(save_dir / "negative_binders.csv", index=False)


def save_unique_per_cow(unique_per_cow: dict, save_dir: Path) -> None:
    """
    Save unique sequences per cow as CSV.

    One row per (cow, sequence).
    """

    rows = []

    for cow, seq_set in unique_per_cow.items():
        for seq in seq_set:
            rows.append({
                "cow": cow,
                "sequence": seq,
                "length": len(seq)
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "unique_per_cow_sequences.csv", index=False)


def save_unique_global_csv(unique_picobody_sequences_global: Dict[str, Dict], save_dir: Path
) -> None:
    """
    Save globally unique sequences with provenance information.
    """

    df = pd.DataFrame([
        {
            "sequence": seq,
            "length": len(seq),
            "occurrence": info.get("occurrence", []),
            "sources": "; ".join(
                f"{cow}/{sample}/{read}"
                for cow, sample, read in info.get("sources", [])
            )
        }
        for seq, info in unique_picobody_sequences_global.items()
    ])

    df.to_csv(save_dir / "unique_sequences.csv", index=False)


def save_raw_sequences(raw_sequences: List[Dict], save_dir: Path
) -> None:
    """
    Save raw FASTA sequences (flattened) as CSV.

    One row per imported sequence.
    """
    df = pd.DataFrame(raw_sequences)
    df.to_csv(save_dir / "picobody_sequences_raw.csv", index=False)


def build_seq_to_id_map(fasta_path: Path) -> dict:
    return {str(r.seq): r.id for r in SeqIO.parse(fasta_path, "fasta")}


def save_candidates(candidate_sequences: Dict[str, Dict], save_dir: Path, seq_to_id: dict) -> None:
    """
    Save candidate sequences as CSV and FASTA.
    """

    df = pd.DataFrame([
        {
            "sequence": seq,
            "length": len(seq),
            "n_sources": len(info.get("sources", []))
        }
        for seq, info in candidate_sequences.items()
    ])

    df.to_csv(save_dir / "candidates.csv", index=False)

    # --- save FASTA ---
    fasta_path = save_dir / "candidates.fasta"

    with fasta_path.open("w") as fh:
        for seq in candidate_sequences:
            seq_id = seq_to_id[seq]
            fh.write(f">{seq_id}\n")
            fh.write(seq + "\n")


def save_unique_global_fasta(unique_global: dict, save_dir: Path) -> Path:
    """
    Save globally unique sequences as FASTA for downstream tools.
    """
    fasta_path = save_dir / "unique_global.fasta"

    with open(fasta_path, "w") as f:
        for i, seq in enumerate(unique_global.keys(), start=1):
            f.write(f">seq_{i}\n{seq}\n")

    return fasta_path


def save_clusters(clusters, path: Path):
    with open(path, "wb") as f:
        pickle.dump(clusters, f)


def load_clusters(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def export_clusters_to_txt(clusters: dict, out_path: Path, truncate_seq: bool = True) -> None:
    """
    Export clusters to a formatted TXT table.

    Parameters
    ----------
    clusters : dict
        Cluster dictionary from clustering pipeline
    out_path : Path
        Output file path
    truncate_seq : bool
        If True, sequences are shortened for readability
    """

    with open(out_path, "w") as f:

        # Header
        f.write(f"{'Cluster':<8} {'ID':<20} {'DisplayID':<10} {'Sequence':<60}\n")
        f.write("=" * 110 + "\n")

        # Data
        for cluster_id, data in sorted(clusters.items()):

            ids = data["ids"]
            seqs = data["sequences"]
            display_ids = data.get("display_ids", {})

            for i, sid in enumerate(ids):

                seq = seqs[i] if i < len(seqs) else "NA"
                disp = display_ids.get(sid, "NA")

                if truncate_seq:
                    seq = seq[:55] + "..." if len(seq) > 55 else seq

                f.write(f"{cluster_id:<8} {sid:<20} {disp:<10} {seq:<60}\n")

    print(f"Cluster table written to: {out_path}")