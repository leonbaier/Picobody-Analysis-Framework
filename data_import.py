from pathlib import Path
from Bio import SeqIO
import pandas as pd
import re # standard lib

def load_fasta_sequences(fasta_base_path: Path) -> dict:
    """
       Load FASTA files from cow/sample/read directory structure.

       Returns:
           picobody_sequences:
           Cow → Sample → Read → list of {id, sequence}
       """

    picobody_sequences = {}
    count_sequences = 0

    filename_pattern = re.compile(r"-(\d+)_R([12])") # regex search for "r"aw string with "-", one or more number, "_R" and "1" or "2"

    for cow_dir in fasta_base_path.iterdir():
        if not cow_dir.is_dir():
            continue

        cow_name = cow_dir.name
        picobody_sequences[cow_name] = {}

        for fasta_file in cow_dir.glob("*.fasta"):
            match = filename_pattern.search(fasta_file.name)
            if not match:
                raise ValueError(f"Unexpected filename format: {fasta_file.name}")

            sample_id = f"Sample{match.group(1)}" # uses S and R from match-regex-search
            read_id = f"Read{match.group(2)}"

            picobody_sequences[cow_name].setdefault(sample_id, {})
            picobody_sequences[cow_name][sample_id].setdefault(read_id, [])

            for record in SeqIO.parse(fasta_file, "fasta"):
                picobody_sequences[cow_name][sample_id][read_id].append(
                    {
                        "id": record.id,
                        # remove trailing gap characters introduced by preprocessing
                        "sequence": str(record.seq).rstrip("-")
                    }
                )
                count_sequences += 1

    print(f"FASTA import completed successfully with {count_sequences} sequences")

    return picobody_sequences


def load_negative_binders(excel_path: Path) -> dict:
    """
    Load experimentally confirmed negative binder sequences from Excel.

    Returns:
        negative_binder_sequences:
        sequence → {name, first_found_in}
    """

    df_negative = pd.read_excel(excel_path, header=2)
    df_negative = df_negative.dropna(subset=["name"])     # remove all rows, where name is nan

    negative_binder_sequences = {}

    for _, row in df_negative.iterrows():
        seq = str(row["Sequence"]).strip() # stores sequence from its row after changing to string in seq and cuts spaces before & after string
        name = row["name"]
        first_found = row["first found in"]
        negative_binder_sequences[seq] = {
                "name": name,
                "first_found_in": first_found
            }

    print(f"Negative-Binder import completed successfully with {len(negative_binder_sequences)} sequences")

    return negative_binder_sequences