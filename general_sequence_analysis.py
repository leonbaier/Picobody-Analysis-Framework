import matplotlib.pyplot as plt
import numpy as np
import logomaker
from Bio import AlignIO

from pathlib import Path
import subprocess, math
from collections import Counter
import time


def plot_length_distribution(unique_global: dict, save_dir: Path = None, filename: str = "unique_global_length_vs_sequence.png"
) -> None:
    """
    Plot length distribution of globally unique sequences.
    """

    lengths = [len(seq) for seq in unique_global.keys()]

    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=20, color="tab:blue", edgecolor="black")
    plt.xlabel("Sequence length (aa)")
    plt.ylabel("Count")
    plt.title("Length distribution of globally unique sequences")
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / filename, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_occurrence_distribution(unique_global: dict, save_dir: Path = None, filename: str = "unique_global_sequence_occurrence.png",
) -> None:
    """
    Plot a histogram of sequence lengths weighted by their occurrence.
    """

    lengths = []
    weights = []

    for sequence, info in unique_global.items():
        lengths.append(len(sequence))
        weights.append(info["occurrence"])

    plt.figure(figsize=(6, 4))
    plt.hist(
        lengths,
        bins=30,
        weights=weights,
        edgecolor="black",
        alpha=0.8,
    )

    plt.xlabel("Sequence length (aa)")
    plt.ylabel("Occurrence (number of observations)")
    plt.title("Occurrence of different sequence lengths")
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / filename, dpi=300)
    else:
        plt.show()

    plt.close()


def run_clustalw_alignment_from_fasta(fasta_files: list[Path], clustalw_path: str, output_dir: Path, output_prefix: str
) -> Path:
    """
    Run ClustalW2 alignment on one or more FASTA files.
    FASTA files are concatenated into a *_prep.fasta file before alignment.
    Final alignment files are renamed to remove '_prep'.
    """

    output_dir.mkdir(exist_ok=True)

    prep_fasta = output_dir / f"{output_prefix}_prep.fasta"
    final_aln = output_dir / f"{output_prefix}.aln"
    final_dnd = output_dir / f"{output_prefix}.dnd"

    # --- combine FASTA files ---
    with open(prep_fasta, "w") as out:
        for fasta in fasta_files:
            with open(fasta) as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"FASTA file is empty: {fasta}")
                out.write(content + "\n")

    print("\nRunning ClustalW alignment...")
    start_time = time.time()

    cmd = [
        clustalw_path,
        f"-INFILE={prep_fasta.name}",
        "-TYPE=PROTEIN",
        "-OUTPUT=CLUSTAL",
        "-OUTORDER=INPUT"]

    subprocess.run(
        cmd,
        cwd=output_dir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)

    # --- rename outputs ---
    prep_aln = output_dir / f"{output_prefix}_prep.aln"
    prep_dnd = output_dir / f"{output_prefix}_prep.dnd"

    if prep_aln.exists():
        if final_aln.exists():
            final_aln.unlink()
        prep_aln.rename(final_aln)
    else:
        raise FileNotFoundError(f"Expected alignment file not found: {prep_aln}")

    if prep_dnd.exists():
        if final_dnd.exists():
            final_dnd.unlink()
        prep_dnd.rename(final_dnd)

    elapsed = time.time() - start_time
    print(f"Alignment finished in {elapsed:.1f} seconds")
    print(f"Alignment written to: {final_aln}")

    return final_aln


def plot_gap_distribution(aln_path: Path, save_dir: Path = None, filename: str | None = None, plot_title: str | None = None
) -> None:
    """
    Plot gap fraction per alignment position.
    At each position: number of gaps divided by number of sequences.
    Long regions with low gap fraction and sharp peaks indicate good conservation.
    """

    if filename is None:
        filename = f"{aln_path.stem}_gap_distribution.png"

    alignment = AlignIO.read(aln_path, "clustal")
    n_seqs = len(alignment)
    aln_len = alignment.get_alignment_length()

    gap_fractions = []
    for pos in range(aln_len):
        column = alignment[:, pos]
        gap_fraction = column.count("-") / n_seqs
        gap_fractions.append(gap_fraction)

    plt.figure(figsize=(10, 4))
    plt.plot(gap_fractions, color="tab:blue")
    plt.xlabel("Alignment position")
    plt.ylabel("Gap fraction")
    if plot_title is None:
        plt.title(f"Gap Distribution of {aln_path.stem} across Alignment")
    else:
        plt.title(plot_title)
    plt.ylim(0, 1)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / filename, dpi=300)
    else:
        plt.show()

    plt.close()


def plot_entropy_distribution(aln_path: Path, save_dir: Path = None, filename: str | None = None, plot_title: str | None = None
) -> None:
    """
    Plot Shannon entropy per alignment position.
        -> At each position: calculate with probability to find a residue at a position by the Shannon entropy formula the entropy.
    Lower value points to higher conservation.
    """

    if filename is None:
        filename = f"{aln_path.stem}_entropy_distribution.png"

    alignment = AlignIO.read(aln_path, "clustal")
    n_seqs = len(alignment)
    aln_len = alignment.get_alignment_length()

    entropies = []
    for pos in range(aln_len):
        column = alignment[:, pos]
        residues = [aa for aa in column if aa != "-"]

        if len(residues) == 0:
            entropies.append(0.0)
            continue

        counts = Counter(residues) # counts number of times each residue occurs
        entropy = 0.0

        for count in counts.values(): # calculates entropy for each residue amount
            p = count / len(residues) # probability of to find this residue at a position
            entropy -= p * math.log2(p)

        entropies.append(entropy)

    plt.figure(figsize=(10, 4))
    plt.plot(entropies, color="tab:blue")
    plt.xlabel("Alignment position")
    plt.ylabel("Shannon entropy")
    if plot_title is None:
        plt.title(f"Entropy Distribution of {aln_path.stem} across Alignment")
    else:
        plt.title(plot_title)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / filename, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_sequence_logo(aln_path: Path, save_dir: Path = None, filename: str | None = None, plot_title: str | None = None, include_gaps: bool = False
) -> None:
    """
    Generate a sequence logo from a ClustalW alignment.
    """

    if filename is None:
        if include_gaps:
            filename = f"{aln_path.stem}_logo_with_gaps.png"
        else:
            filename = f"{aln_path.stem}_logo_no_gaps.png"

    alignment = AlignIO.read(aln_path, "clustal")
    logo_sequences = []
    for record in alignment:
        logo_sequences.append(str(record.seq))

    counts = logomaker.alignment_to_matrix( # counts number of times each residue occurs at each position
        logo_sequences,
        to_type="counts",
        characters_to_ignore="-")

    if include_gaps:
        n_seqs = len(logo_sequences)
        freq = counts / n_seqs
    else:
        freq = counts.div(counts.sum(axis=1), axis=0) # creates freq matrix by dividing each aa count by the sum of all aa at this position

    plt.figure(figsize=(12, 4))
    logo = logomaker.Logo(
        freq,
        color_scheme="chemistry")

    if include_gaps and "-" in freq.columns:
        logo.style_glyphs(
            glyphs="-",
            color="lightgray"
        )

    plt.xlabel("Alignment position")
    plt.ylabel("Residue frequency")
    if plot_title is None:
        plt.title(f"Sequence Logo of {aln_path.stem}")
    else:
        plt.title(plot_title)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir / filename, dpi=300)
    else:
        plt.show()
    plt.close()


