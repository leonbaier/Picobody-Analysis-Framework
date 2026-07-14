from typing import Dict, List, Tuple, Any
from Bio import SeqIO
from pathlib import Path

def deduplicate_per_cow(picobody_sequences: Dict
     ) -> Dict[Dict, Dict]:
    """
    Collect unique amino acid sequences per cow.
    """

    unique_per_cow = {}
    for cow, samples in picobody_sequences.items():
        seq_set = set()

        for sample in samples.values():
            for read in sample.values():
                for entry in read:
                    seq_set.add(entry["sequence"])

        unique_per_cow[cow] = seq_set

    return unique_per_cow


def deduplicate_global(picobody_sequences: Dict
) -> Dict[str, Dict]:
    """
    Deduplicate sequences globally across all cows.

    Returns:
        unique_picobody_sequences_global:
        sequence → {id, sources[(cow, sample, read)], occurrence}
    """

    unique_picobody_sequences_global = {}
    for cow, samples in picobody_sequences.items():
        for sample, reads in samples.items():
            for read, seq_list in reads.items():
                for entry in seq_list:
                    seq = entry["sequence"]
                    seq_id = entry["id"]

                    if seq not in unique_picobody_sequences_global:
                        unique_picobody_sequences_global[seq] = {
                            "id": seq_id,
                            "sources": [(cow, sample, read)],
                            "occurrence": 1
                        }
                    else:
                        unique_picobody_sequences_global[seq]["sources"].append(
                            (cow, sample, read)
                        )
                        unique_picobody_sequences_global[seq]["occurrence"] += 1

    return unique_picobody_sequences_global


def flatten_raw_sequences(picobody_sequences: Dict
) -> List[Dict]:
    """
    Flatten nested raw FASTA structure into a list of records.
    """

    raw_sequences = []
    for cow, samples in picobody_sequences.items():
        for sample, reads in samples.items():
            for read, seq_list in reads.items():
                for entry in seq_list:
                    raw_sequences.append({
                        "cow": cow,
                        "sample": sample,
                        "read": read,
                        "id": entry["id"],
                        "sequence": entry["sequence"]
                    })

    return raw_sequences


def check_negative_occurrence(raw_sequences: List[Dict], unique_global: Dict[str, Dict], negative_binder_sequences: Dict[str, Dict]
) -> Dict[str, set]:
    """
    Check where negative binder sequences occur:
    - in raw FASTA data
    - in global unique sequences
    """

    raw_seq_set = set() # set uses every element only one time (deduplicates)
    for entry in raw_sequences:
        raw_seq_set.add(entry["sequence"])

    neg_in_raw = set()
    for seq in negative_binder_sequences:
        if seq in raw_seq_set:
            neg_in_raw.add(seq)

    neg_in_global = set()
    for seq in negative_binder_sequences:
        if seq in unique_global:
            neg_in_global.add(seq)

    neg_not_in_global = set()
    for seq in negative_binder_sequences:
        if seq not in unique_global:
            neg_not_in_global.add(seq)

    return {
        "neg_in_raw": neg_in_raw,
        "neg_in_global": neg_in_global,
        "neg_not_in_global": neg_not_in_global,
    }


def extract_negative_binder_ids_from_fasta(unique_global_fasta: Path, negative_binder_sequences: Dict[str, Dict],
) -> Dict[str, Dict[str, str]]:
    """
    Map negative binder sequences to canonical IDs using unique_global.fasta.
    """

    # --- read unique global FASTA ---
    seq_to_id: Dict[str, str] = {}
    for record in SeqIO.parse(unique_global_fasta, "fasta"):
        seq_to_id[str(record.seq)] = record.id

    negative_binders: Dict[str, Dict[str, str]] = {}

    # --- find negative binders in FASTA ---
    for neg_seq, meta in negative_binder_sequences.items():
        if neg_seq in seq_to_id:
            fasta_id = seq_to_id[neg_seq]
            negative_binders[fasta_id] = {
                "sequence": neg_seq,
                "source": meta}
            print(
                f"Found negative binder → ID: {fasta_id} | "
                f"Name: {meta.get('name', 'N/A')} | "
                f"Sequence: {neg_seq}")

    return negative_binders


def build_candidate_set(unique_global, negative_binder_ids, seq_to_id):

    candidate_sequences = {}

    for seq, info in unique_global.items():

        seq_id = seq_to_id.get(seq)

        if seq_id not in negative_binder_ids:
            candidate_sequences[seq] = info

    return candidate_sequences