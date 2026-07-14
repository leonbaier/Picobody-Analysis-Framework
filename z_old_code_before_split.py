# imports
from pathlib import Path
from Bio import SeqIO
import pandas as pd
import re # standard lib
import pickle


# paths
fasta_base_path = Path(
    r"\\nas.ads.mwn.de\ge63laz\TUM-PC\Desktop\Masterthesis_Picobodies\Source_Structure Thesis\Picobody_Sequences_Anti-mClover\Fasta-Files")
excel_path = Path(
    r"\\nas.ads.mwn.de\ge63laz\TUM-PC\Desktop\Masterthesis_Picobodies\Source_Structure Thesis\Picobody_Sequences_Anti-mClover\Overview_Sequences_Picobody_anti-mClover.xlsx")


# read fasta
picobody_sequences = {}
count_sequences=0
filename_pattern = re.compile(r"-(\d+)_R([12])") # regex search for "r"aw string with "-", one or more number, "_R" and "1" or "2"

for cow_dir in fasta_base_path.iterdir(): # for both cow folder, read subfolder
    if not cow_dir.is_dir(): # if other data than folder are stored there
        continue
    cow_name = cow_dir.name
    picobody_sequences[cow_name] = {}

    for fasta_file in cow_dir.glob("*.fasta"): # searching all fasta data in it
        match = filename_pattern.search(fasta_file.name) # search for name in file name, with match either success or None
        if not match:
            raise ValueError(f"Unexpected filename format: {fasta_file.name}")
        sample_id = f"Sample{match.group(1)}" # uses S and R from match-regex-search
        read_id = f"Read{match.group(2)}"

        picobody_sequences[cow_name].setdefault(sample_id, {}) # creates S and R keys if not existing
        picobody_sequences[cow_name][sample_id].setdefault(read_id, [])

        for record in SeqIO.parse(fasta_file, "fasta"): # read every sequence in fasta
            picobody_sequences[cow_name][sample_id][read_id].append(
                {
                    "id": record.id,
                    "sequence": str(record.seq).rstrip("-")
                }
            )
            count_sequences += 1

print(f"FASTA import completed successfully with {count_sequences} sequences")
print("-------------------------------------------")


# filter unique sequences per cow
unique_picobody_sequences_per_cow = {}
unique_sequence_counts_per_cow = {}

for cow, samples in picobody_sequences.items():  # for each cow each sample
    unique_picobody_sequences_per_cow[cow] = {}
    unique_sequence_counts_per_cow[cow] = 0

    for sample, reads in samples.items():  # for each sample each read
        for read, seq_list in reads.items():  # for each read the sequence list
            for entry in seq_list:
                seq = entry["sequence"]
                seq_id = entry["id"]

                if seq not in unique_picobody_sequences_per_cow[cow]:
                    unique_picobody_sequences_per_cow[cow][seq] = {
                        "id": seq_id,
                        "sources": [(sample, read)]
                    }
                    unique_sequence_counts_per_cow[cow] += 1
                else:
                    unique_picobody_sequences_per_cow[cow][seq]["sources"].append(
                        (sample, read)
                    )

for cow, count in unique_sequence_counts_per_cow.items():
    print(f"{cow}: {count} unique sequences")


# filter unique sequences global
unique_picobody_sequences_global = {}
count_unique_sequences_global = 0

for cow, seq_dict in unique_picobody_sequences_per_cow.items(): # for each cow each sequence data
    for seq, info in seq_dict.items(): # for each sequence and each info
        if seq not in unique_picobody_sequences_global:
            unique_picobody_sequences_global[seq] = {
                "id": info["id"],
                "sources": [(cow, *src) for src in info["sources"]] # for every existing source: adds in source info cow name + existing source info
            }
            count_unique_sequences_global += 1
        else:
            # append additional sources (from another cow or sample)
            unique_picobody_sequences_global[seq]["sources"].extend(
                [(cow, *src) for src in info["sources"]]
            )

print(f"General: {count_unique_sequences_global} unique sequences")

''' Test
for seq, info in unique_picobody_sequences_global.items():
    print(seq, info["sources"])
'''


# read negative binders from excel
df_negative = pd.read_excel(excel_path, header=2)
df_negative = df_negative.dropna(subset=["name"]) # delete complete row when name is nan

negative_binder_sequences = {}
for _, row in df_negative.iterrows():
    seq = str(row["Sequence"]).strip() # stores sequence from its row in seq and cuts spaces before & after string
    name = row["name"]
    first_found = row["first found in"]

    negative_binder_sequences[seq] = {
        "name": name,
        "first_found_in": first_found
    }

print(f"Imported {len(negative_binder_sequences)} negative binder sequences")


# sorting in potential binder (candidates) and negative ones
candidate_sequences = {
    seq: info
    for seq, info in unique_picobody_sequences_global.items()
    if seq not in negative_binder_sequences
}
print(f"Number of potential binders: {len(candidate_sequences)}")
print("-------------------------------------------")


# check for occurrence of negative binders in given sequences
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
raw_seq_set = {entry["sequence"] for entry in raw_sequences}
print("Negatives present in raw data:",
      len(raw_seq_set & set(negative_binder_sequences)))

neg_in_global = {
    seq for seq in negative_binder_sequences
    if seq in unique_picobody_sequences_global
}
neg_not_in_global = {
    seq for seq in negative_binder_sequences
    if seq not in unique_picobody_sequences_global
}
print("Negative in global unique:", len(neg_in_global))
print("Negative NOT in global unique:", len(neg_not_in_global))
print("-------------------------------------------")




# save sequences
save_dir = Path("data_saved") # data path
save_dir.mkdir(exist_ok=True)

# safe all sequences (csv)
df_raw = pd.DataFrame(raw_sequences)
df_raw.to_csv(
    save_dir / "picobody_sequences_raw.csv",
    index=False
)

# safe unique sequences global (csv)
df_unique_global = pd.DataFrame([
    {
        "sequence": seq,
        "length": len(seq),
        "n_sources": len(info.get("sources", [])),
        "sources": "; ".join(
            f"{cow}/{sample}/{read}"
            for cow, sample, read in info.get("sources", [])
        )
    }
    for seq, info in unique_picobody_sequences_global.items()
])
df_unique_global.to_csv(
    save_dir / "unique_global_sequences.csv",
    index=False
)

# safe negative binders (csv)
df_neg = pd.DataFrame([
    {
        "sequence": seq,
        "name": info["name"]
    }
    for seq, info in negative_binder_sequences.items()
])
df_neg.to_csv(save_dir / "negative_binders.csv", index=False)

# safe candidates (csv, pickle)
df_candidates = pd.DataFrame([
    {
        "sequence": seq,
        "length": len(seq),
        "n_sources": len(info.get("sources", []))
    }
    for seq, info in candidate_sequences.items()
])
df_candidates.to_csv(save_dir / "candidates.csv", index=False)

with open(save_dir / "candidates.pkl", "wb") as f:
    pickle.dump(candidate_sequences, f)
