# not part of this project, alone on server
from pathlib import Path
import torch
import esm
from Bio import SeqIO
import re
import numpy as np
import time

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=\-]", "_", s) # inname


# -------------------------
# Paths
# -------------------------
FASTA_PATH = Path("esm_input.fasta")
OUT_DIR = Path("pdb_out")

OUT_DIR.mkdir(parents=True, exist_ok=True)
start_time_main = time.time()

# -------------------------
# Load Model
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # gpu or cpu

print("Loading ESMFold model...")
model = esm.pretrained.esmfold_v1() # loads ~ 3 Gb model
model = model.eval().to(DEVICE)  # eval() turns training mode off (-> inference mode) and model is shifted to device (gpu/cpu)
model.set_chunk_size(64)

# -------------------------
# process FASTA
# -------------------------
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    seq_id = safe_filename(record.id)
    seq = str(record.seq)

    if len(seq) == 0:
        print(f"Skipping empty sequence: {seq_id}")
        continue

    print(f"Folding {seq_id} (length={len(seq)})")

    with torch.no_grad():  # deactivates learning and gradient storing: faster and memory-efficient
        pdb_str = model.infer_pdb(seq)
        output = model.infer(seq) # folding itself

    # --- PDB ---
    (OUT_DIR / f"{seq_id}.pdb").write_text(pdb_str)

    # --- pLDDT ---
    plddt = output["plddt"].cpu().numpy()
    np.save(OUT_DIR / f"{seq_id}_plddt.npy", plddt)

    # --- PAE (optional, version-dependent) ---
    if "pae" in output:
        pae = output["pae"].cpu().numpy()
        np.save(OUT_DIR / f"{seq_id}_pae.npy", pae)
    else:
        print(f"No PAE available for {seq_id} (ESMFold version limitation)")

end_time_main = time.time()
elapsed_main = end_time_main - start_time_main
print(f"All sequences folded with a total runtime of {elapsed_main:.1f} seconds.")