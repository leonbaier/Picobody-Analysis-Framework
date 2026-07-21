from pathlib import Path
import re
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch
from Bio import SeqIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser


def get_negative_binder_ids_from_vx_name(excel_path: Path, binders: str | list[str],
) -> dict[str, str]:
    """
    Return internal seq IDs for negative binders.

    Example:
    >>> get_negative_binder_ids_from_vx_name(Path("clustering_summary.xlsx"), ["v1", "v5"])

    {
        "v1": "seq_25",
        "v5": "seq_28"
    }
    """

    if isinstance(binders, str):
        binders = [binders]

    df = pd.read_excel(excel_path, sheet_name="Sequences")
    negatives = df[df["Type"] == "negative"]
    result = {}

    for binder in binders:
        target_name = f"anti-mClover-ulCDR-{binder}-Fd"
        hit = negatives[negatives["Original_Name"] == target_name]

        if not hit.empty:
            result[binder] = hit.iloc[0]["ID"]

    return result


def build_comparison_and_tested_id_sets(comparison_ids: list[str], tested_ids: list[str],
        additional_comparison_ids: list[str] | None = None, additional_tested_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Build two sorted ID sets.

    Example
    -------
    comparison_ids:
        ["seq_28", "seq_39"]

    additional_comparison_ids:
        ["seq_5"]

    ->
        ["seq_5", "seq_28", "seq_39"]
    """

    additional_comparison_ids = additional_comparison_ids or []
    additional_tested_ids = additional_tested_ids or []

    comparison_result = sorted(
        set(comparison_ids + additional_comparison_ids),
        key=lambda x: int(re.search(r"\d+", x).group()))

    tested_result = sorted(
        set(tested_ids + additional_tested_ids),
        key=lambda x: int(re.search(r"\d+", x).group()))

    return comparison_result, tested_result


def build_global_display_order(clusters: dict) -> dict:

    display_index = {}

    for cluster_data in clusters.values():

        display_index.update(
            cluster_data["display_ids"]
        )

    return display_index


def collect_plddt_stats(plddt_dir: Path) -> dict:
    """
    Collect pLDDT statistics for all models in a directory.
    """
    results = {}
    for plddt_file in plddt_dir.glob("*_plddt.npy"):

        # FIX: skip chain files
        if "_chainA" in plddt_file.name:
            continue
        model_id = plddt_file.stem.replace("_plddt", "")

        plddt = np.load(plddt_file)
        # Residue reduzieren
        if plddt.ndim == 3:
            # (1, N_res, 37)
            plddt = plddt.squeeze(0)  # → (N_res, 37)
            plddt = plddt.mean(axis=-1)  # → (N_res,)
        elif plddt.ndim == 2:
            # (N_res, 37)
            plddt = plddt.mean(axis=-1)  # → (N_res,)
        plddt = np.squeeze(plddt)


        results[model_id] = {
            "mean": float(plddt.mean()),
            "median": float(np.median(plddt)),
            "fraction_below_50": float((plddt < 50).mean()),
            "fraction_above_70": float((plddt > 70).mean()),
            "plddt_path": plddt_file,
        }

    """
    # debug
    for f in plddt_dir.glob("*_plddt.npy"):
        x = np.load(f)
        print(f.name, x.shape)
    """

    return results


def get_max_residue_length(plddt_stats: dict) -> int:
    """
    Get maximal residue length across all models.
    """

    max_len = 0

    for stats in plddt_stats.values():

        arr = np.load(stats["plddt_path"])

        if isinstance(arr, np.ndarray):
            plddt = np.squeeze(arr)
        else:
            plddt = np.squeeze(arr[arr.files[0]])

        current_len = plddt.shape[-1] if plddt.ndim > 1 else len(plddt)

        max_len = max(max_len, current_len)

    return max_len


def build_model_to_cluster_from_fasta(fasta_path: Path) -> dict[str, int]:
    """
    Build a mapping from sanitized model IDs to cluster indices
    based on FASTA headers of the form:
    >cluster_<N>_<rest>
    """

    model_to_cluster = {}
    cluster_pattern = re.compile(r"^cluster_(\d+)_")

    for record in SeqIO.parse(fasta_path, "fasta"):
        raw_id = record.id
        sanitized_id = raw_id.replace("|", "_")

        match = cluster_pattern.match(raw_id)
        if match is None:
            raise ValueError(f"Invalid FASTA header: {raw_id}")

        cluster_id = int(match.group(1))
        model_to_cluster[sanitized_id] = cluster_id

    return model_to_cluster


def normalize_model_id(mid: str) -> str:
    """
    Normalize model IDs to match FASTA headers.
    """

    # Remove ligand suffixes
    suffixes = [
        "_with_ligand",
        "_without_ligand",
        "_ligand",
    ]
    for suffix in suffixes:
        mid = mid.replace(suffix, "")
    mid = re.sub(r"_chain[A-Za-z0-9]+", "", mid)

    return mid


def plot_plddt_landscape(plddt_stats: dict, model_to_cluster: dict, save_path=None, display_index=None, max_residue_len=None, model_name=None
):
    """
    Plot pLDDT landscape with:
    - cluster annotation (left)
    - residue-wise pLDDT heatmap (center)
    - mean pLDDT per structure (right)
    """

    def get_cluster(mid):
        for key in model_to_cluster:
            if key == mid:
                return model_to_cluster[key]
        return -1


    if display_index is not None:
        model_ids = sorted(
            plddt_stats.keys(),
            key=lambda mid: (
                get_cluster(mid),
                display_index.get(extract_seq_id(mid), 0)
            )
        )
    y_labels = []
    for mid in model_ids:
        seq_id = extract_seq_id(mid)
        y_labels.append(display_index.get(seq_id, 0))


    # --- load and reduce pLDDT arrays ---
    plddt_arrays = []
    mean_plddt = []

    for mid in model_ids:
        data = np.load(plddt_stats[mid]["plddt_path"])

        if isinstance(data, np.ndarray):
            plddt = data
        else:
            plddt = data[data.files[0]]

        # Reduce to residue-level pLDDT
        if plddt.ndim == 3:
            plddt = plddt.squeeze()  # (N_res, 37)
            plddt = plddt.mean(axis=-1)  # → (N_res,)

        elif plddt.ndim == 2:
            plddt = plddt.mean(axis=-1)  # → (N_res,)
        plddt = np.squeeze(plddt)

        plddt_arrays.append(plddt)
        mean_plddt.append(plddt.mean())

    data_max_len = max(len(a) for a in plddt_arrays)

    max_len = data_max_len

    # Build padded matrix
    heatmap = np.full((len(plddt_arrays), max_len), np.nan)
    for i, arr in enumerate(plddt_arrays):
        heatmap[i, :len(arr)] = arr

    # Cluster vector
    cluster_ids = np.array([
        model_to_cluster.get(normalize_model_id(mid), -1)
        for mid in model_ids
    ])

    # --- figure layout ---
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=3,
        width_ratios=[0.3, 4, 0.7],  # mean panel narrower
        wspace=0.03  # slightly tighter spacing
    )

    ax_cluster = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1], sharey=ax_cluster)
    ax_mean = fig.add_subplot(gs[2], sharey=ax_cluster)

    # Manually shift the mean plot slightly to the left
    pos = ax_mean.get_position()
    ax_mean.set_position((
        pos.x0 + 0.05,  # shift manually mean plot in plot to the right
        pos.y0,
        pos.width,
        pos.height))

    # --- cluster strip (left) ---
    cluster_img = cluster_ids[:, None]
    norm = Normalize(vmin=0, vmax=17)
    im_cluster = ax_cluster.imshow(
        cluster_img,
        aspect="auto",
        cmap="tab20",
        norm=norm,
        interpolation="nearest")

    ax_cluster.set_xticks([])
    ax_cluster.set_ylabel("Structure index")

    if display_index is not None:

        n_models = len(model_ids)
        all_positions = np.arange(n_models)

        major_ticks = []
        minor_ticks = []
        labels = [""] * n_models

        for i in range(n_models):

            val = y_labels[i]

            # --- Hauptticks: 10er ---
            if val % 10 == 0:
                major_ticks.append(i)
                labels[i] = str(val)

            # --- Nebenticks: 5er ---
            elif val % 5 == 0:
                minor_ticks.append(i)

        # --- set main ticks ---
        ax_cluster.set_yticks(major_ticks)
        ax_cluster.set_yticklabels([labels[i] for i in major_ticks])

        # --- set minor ticks ---
        ax_cluster.set_yticks(minor_ticks, minor=True)

        # --- Tick-Style ---
        ax_cluster.tick_params(axis="y", which="major", length=6, width=1.2)
        ax_cluster.tick_params(axis="y", which="minor", length=4, width=2.0)

    ax_cluster.set_title("Cluster")

    # --- pLDDT heatmap (center) ---
    im = ax_heatmap.imshow(
        heatmap,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )
    ax_heatmap.set_xlim(0, max_len)
    ax_heatmap.set_xlabel("Residue position")
    ax_heatmap.set_title("Residue-wise pLDDT")

    # --- mean pLDDT bars (right) ---
    ax_mean.barh(
        np.arange(len(mean_plddt)),
        mean_plddt,
        color="black",
        alpha=0.6,
    )
    ax_mean.axvline(70, color="red", linestyle="--", linewidth=1)
    ax_mean.set_xlabel("Mean pLDDT")
    ax_mean.set_xlim(0, 100)
    ax_mean.set_title("Mean")

    # Clean shared y-axis clutter
    plt.setp(ax_heatmap.get_yticklabels(), visible=False)
    plt.setp(ax_mean.get_yticklabels(), visible=False)

    # --- colorbar ---
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.01)
    cbar.set_label("pLDDT")

    if model_name is not None:
        title = (
            f"{model_name} pLDDT landscape "
            f"with cluster annotation and per-model mean")
    else:
        title = (
            "pLDDT landscape "
            "with cluster annotation and per-model mean")

    fig.suptitle(
        title,
        fontsize=14,
        y=0.98)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"plddt-Plot written to: {save_path}")
    else:
        plt.show()

    plt.close()


def collect_boltz_best_models(base_pred_dir: Path):
    """
    Loop over all prediction folders and select best model.

    Returns plddt_stats dict with:
        - mean
        - plddt_path
        - best_model name
        - cif_path
    """

    plddt_stats = {}

    for pred_dir in base_pred_dir.iterdir():
        if not pred_dir.is_dir():
            continue

        best_score = -1
        best_plddt = None
        best_model = None
        best_cif = None

        # --- find best model ---
        for npz_file in pred_dir.glob("plddt_*.npz"):

            data = np.load(npz_file)
            plddt = data[data.files[0]]
            plddt = np.asarray(plddt)

            while plddt.ndim > 1:
                plddt = plddt.mean(axis=-1)

            plddt = np.squeeze(plddt)
            plddt = plddt * 100

            score = plddt.mean()

            if score > best_score:
                best_score = score
                best_plddt = plddt
                best_model = npz_file.stem.replace("plddt_", "")
                best_cif = pred_dir / f"{best_model}.cif"

        if best_plddt is None:
            continue

        model_id = pred_dir.name
        out_path = pred_dir / f"{model_id}_plddt.npy"
        np.save(out_path, best_plddt)

        plddt_stats[model_id] = {
            "mean": float(best_score),
            "plddt_path": out_path,
            "model": best_model,
            "cif_path": best_cif,
        }

    return plddt_stats


def collect_af_best_models(base_pred_dir: Path):
    """
    Collect best AF3 models and convert atom-wise pLDDT
    to residue-wise pLDDT using MMCIF atom mapping (cif data in folder necessary!).

    Returns:
        plddt_stats dict compatible with plotting functions.
    """

    plddt_stats = {}

    parser = MMCIFParser(QUIET=True)

    for pred_dir in base_pred_dir.iterdir():

        if not pred_dir.is_dir():
            continue

        if pred_dir.name in ["msas", "templates"]:
            continue

        best_score = -1
        best_plddt = None
        best_model = None
        best_cif = None

        # --- iterate over AF3 full_data files ---
        for json_file in pred_dir.glob("*full_data_*.json"):

            with open(json_file) as fh:
                data = json.load(fh)

            # --- atom-wise pLDDT from AF3 ---
            atom_plddt = np.array(data["atom_plddts"])

            # --- corresponding CIF path ---
            model_idx = json_file.stem.split("_")[-1]

            # remove "_full_data_<idx>"
            prefix = json_file.stem.rsplit("_full_data_", 1)[0]

            cif_path = pred_dir / (
                f"{prefix}_model_{model_idx}.cif"
            )

            if not cif_path.exists():
                print(f"Missing CIF file: {cif_path}")
                continue

            # --- parse structure ---
            structure = parser.get_structure(
                "model",
                str(cif_path)
            )

            # --- collect residue-wise values ---
            residue_to_scores = defaultdict(list)

            atom_counter = 0

            for atom in structure.get_atoms():

                # skip hydrogens
                if atom.element == "H":
                    continue

                residue = atom.get_parent()

                # skip waters
                if residue.id[0] == "W":
                    continue

                # avoid overflow protection
                if atom_counter >= len(atom_plddt):
                    break

                # unique residue identifier
                res_key = (
                    residue.get_parent().id,   # chain id
                    residue.id[1]              # residue number
                )

                residue_to_scores[res_key].append(
                    atom_plddt[atom_counter]
                )

                atom_counter += 1

            # --- sanity check ---
            assert atom_counter == len(atom_plddt), (
                f"Atom count mismatch in {json_file.name}: "
                f"{atom_counter} CIF atoms vs "
                f"{len(atom_plddt)} pLDDT values"
            )

            # --- residue-wise mean pLDDT ---
            plddt = np.array([
                np.mean(scores)
                for scores in residue_to_scores.values()
            ])

            score = plddt.mean()

            # --- keep best model ---
            if score > best_score:

                best_score = score
                best_plddt = plddt

                best_model = f"model_{model_idx}"

                best_cif = cif_path

        # --- skip empty entries ---
        if best_plddt is None:
            continue

        model_id = pred_dir.name

        out_path = pred_dir / f"{model_id}_plddt.npy"

        np.save(out_path, best_plddt)

        plddt_stats[model_id] = {
            "mean": float(best_score),
            "plddt_path": out_path,
            "model": best_model,
            "cif_path": best_cif,
        }

    return plddt_stats


def build_af_model_to_cluster(plddt_stats: dict, clusters: dict) -> dict:
    """
    Build AF3 model → cluster mapping based on sequence IDs.

    Example:
        seq1_af3 -> cluster 3
    """

    # --- build seq_id -> cluster lookup ---
    seq_to_cluster = {}

    for cluster_id, cluster_data in clusters.items():

        for seq_id in cluster_data["ids"]:

            # normalize IDs
            seq_clean = seq_id.replace("_", "").lower()

            seq_to_cluster[seq_clean] = cluster_id

    model_to_cluster = {}

    for mid in plddt_stats:

        # example:
        # seq1_af3 -> seq1
        match = re.match(r"(seq\d+)", mid.lower())

        if match is None:
            raise ValueError(f"Could not parse AF3 model id: {mid}")

        seq_id = match.group(1)

        if seq_id not in seq_to_cluster:
            raise ValueError(f"{seq_id} not found in clusters")

        model_to_cluster[mid] = seq_to_cluster[seq_id]

    return model_to_cluster


def extract_seq_id(mid: str) -> str:

    # seq IDs
    match = re.search(r"orig_id=(seq_\d+)", mid)
    if match:
        return match.group(1)

    # knob IDs
    match = re.search(r"orig_id=(knob_\d+)", mid)
    if match:
        return match.group(1)

    # fallback (Boltz etc.)
    match = re.search(r"(knob_\d+)", mid)
    if match:
        return match.group(1)

    match = re.search(r"(seq_\d+)", mid)
    if match:
        return match.group(1)

    match = re.search(r"seq_?(\d+)", mid)
    if match:
        return f"seq_{match.group(1)}"

    raise ValueError(f"Cannot parse seq_id from {mid}")


def filter_plddt_stats_by_ids(plddt_stats: dict, wanted_ids: list[str],
) -> dict:

    wanted_ids = set(wanted_ids)

    return {
        mid: stats
        for mid, stats in plddt_stats.items()
        if extract_seq_id(mid) in wanted_ids
    }


def load_plddt_array(plddt_path):
    data = np.load(plddt_path)

    if isinstance(data, np.ndarray):
        plddt = data
    else:
        plddt = data[data.files[0]]

    if plddt.ndim == 3:
        plddt = plddt.squeeze()
        plddt = plddt.mean(axis=-1)

    elif plddt.ndim == 2:
        plddt = plddt.mean(axis=-1)

    return np.squeeze(plddt)


def plot_plddt_landscape_groups(stats_without, stats_chainA, tested_ids, comparison_ids, save_path=None, model_name=None,
):
    ordered_rows = []

    # all without ligand first
    for seq_id in tested_ids:
        ordered_rows.append((f"{seq_id} -L",
            next(
                v for k, v in stats_without.items()
                if extract_seq_id(k) == seq_id
            ), "tested"
        ))

    for seq_id in comparison_ids:
        ordered_rows.append((f"{seq_id} -L",
            next(
                v for k, v in stats_without.items()
                if extract_seq_id(k) == seq_id
            ), "comparison"
        ))

    # spacer row
    ordered_rows.append((" ", None, "spacer"))

    # all chain A afterwards
    for seq_id in tested_ids:
        ordered_rows.append((f"{seq_id} +L",
            next(
                v for k, v in stats_chainA.items()
                if extract_seq_id(k) == seq_id
            ), "tested"
        ))

    for seq_id in comparison_ids:
        ordered_rows.append((f"{seq_id} +L",
            next(
                v for k, v in stats_chainA.items()
                if extract_seq_id(k) == seq_id
            ), "comparison"
        ))

    group_ids = []
    labels = []
    plddt_arrays = []
    mean_plddt = []

    for label, stats, group in ordered_rows:
        labels.append(label)
        if group == "tested":
            group_ids.append(0)

        elif group == "comparison":
            group_ids.append(1)

        else:
            group_ids.append(np.nan)

        if stats is None:
            plddt_arrays.append(np.array([]))
            mean_plddt.append(np.nan)
            continue
        arr = load_plddt_array(stats["plddt_path"])
        plddt_arrays.append(arr)
        mean_plddt.append(arr.mean())

    max_len = max(len(x) for x in plddt_arrays)
    heatmap = np.full((len(plddt_arrays), max_len), np.nan)

    for i, arr in enumerate(plddt_arrays):
        heatmap[i, :len(arr)] = arr

    group_img = np.array(group_ids)[:, None]
    group_img = np.ma.masked_invalid(group_img)
    group_cmap = ListedColormap([
        "royalblue",  # tested
        "firebrick",  # comparison
    ])
    group_cmap.set_bad("white")

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=3,
        width_ratios=[0.3, 4, 0.7],  # mean panel narrower
        wspace=0.03  # slightly tighter spacing
    )

    ax_group = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1], sharey=ax_group)
    ax_mean = fig.add_subplot(gs[2], sharey=ax_group)

    pos = ax_mean.get_position()
    ax_mean.set_position((pos.x0 + 0.05, pos.y0, pos.width, pos.height,))

    # group plot
    ax_group.imshow(
        group_img,
        aspect="auto",
        cmap=group_cmap,
        interpolation="nearest",)

    tick_positions = [
        i for i, label in enumerate(labels)
        if label.strip()]
    tick_labels = [
        label for label in labels
        if label.strip()]
    ax_group.set_yticks(tick_positions)
    ax_group.set_yticklabels(tick_labels, fontsize=8,)
    ax_group.tick_params(axis="y", which="major", length=6, width=1.2,)
    ax_group.yaxis.tick_left()

    ax_group.set_xticks([])

    legend_handles = [
        Patch(color="royalblue", label="Tested"),
        Patch(color="firebrick", label="Comparison"),]
    ax_group.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        fontsize=8,
        frameon=False,)

    # heatmap
    im = ax_heatmap.imshow(
        heatmap,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=100,
        interpolation="nearest",)

    plt.setp(ax_heatmap.get_yticklabels(), visible=False,)
    ax_heatmap.tick_params(
        axis="y",
        left=False,
        labelleft=False,)
    ax_heatmap.set_xlabel("Residue position")
    ax_heatmap.set_title("Residue-wise pLDDT")

    # mean plot
    ax_mean.barh(
        np.arange(len(mean_plddt)),
        mean_plddt,
        color="black",
        alpha=0.6,)

    ax_mean.axvline(70, color="red", linestyle="--")

    ax_mean.set_xlim(0, 100)
    ax_mean.set_xlabel("Mean pLDDT")
    ax_mean.set_title("Mean")

    plt.setp(ax_mean.get_yticklabels(), visible=False)
    ax_mean.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
    )

    ax_group.set_ylim(ax_heatmap.get_ylim())
    ax_mean.set_ylim(ax_heatmap.get_ylim())

    cbar = fig.colorbar(
        im,
        ax=ax_heatmap,
        fraction=0.046,
        pad=0.01,)

    cbar.set_label("pLDDT")
    fig.suptitle(f"{model_name}: tested vs comparison", fontsize=14,)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",)
    else:
        plt.show()

    plt.close()


def plot_mean_plddt_multi_models(plddt_stats_dict: dict, labels: list[str], save_path=None, display_index=None,
                                 max_structure_index=None, title=None,
):
    """
    Compare mean pLDDT across models (1:1 mapping, no aggregation).
    """

    plt.figure(figsize=(10, 4))

    for label, stats in plddt_stats_dict.items():

        x_vals = []
        y_vals = []

        if display_index is not None:
            model_ids = sorted(
                stats.keys(),
                key=lambda mid: display_index[extract_seq_id(mid)]
            )
        else:
            model_ids = sorted(stats.keys())

        for mid in model_ids:

            data = stats[mid]
            seq_id = extract_seq_id(mid)

            if display_index is not None:
                if seq_id not in display_index:
                    continue
                x = display_index[seq_id]
            else:
                x = len(x_vals)

            y = data["mean"]

            x_vals.append(x)
            y_vals.append(y)

        plt.scatter(x_vals, y_vals, label=label, alpha=0.7, s=20)

    plt.axhline(70, color="red", linestyle="--")

    if max_structure_index is not None:
        plt.xlim(0, max_structure_index)

    plt.ylim(0, 100)
    plt.xlabel("Structure index")
    plt.ylabel("Mean pLDDT")

    if title:
        plt.title(title)
    else:
        plt.title("Mean pLDDT comparison")

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Comparison plot written to: {save_path}")
    else:
        plt.show()

    plt.close()


def collect_esm_chain_models(base_dir: Path):

    plddt_stats = {}

    for npy_file in base_dir.glob("*_plddt.npy"):

        model_id = npy_file.stem.replace("_plddt", "")

        pdb_file = base_dir / f"{model_id}.pdb"
        if not pdb_file.exists():
            continue

        plddt = np.load(npy_file)

        if plddt.ndim == 3:
            plddt = plddt.squeeze().mean(axis=-1)
        elif plddt.ndim == 2:
            plddt = plddt.mean(axis=-1)

        # --- extract chain A based on structure (correct approach) ---
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("model", str(pdb_file))

        # reduce to residue-level
        while plddt.ndim > 1:
            plddt = plddt.mean(axis=-1)

        plddt = np.squeeze(plddt)

        selected_scores = []
        residue_counter = 0

        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] != " ":
                        continue

                    if residue_counter >= len(plddt):
                        break

                    if chain.id == "A":
                        selected_scores.append(plddt[residue_counter])

                    residue_counter += 1

        plddt_chain = np.array(selected_scores)

        out_path = base_dir / f"{model_id}_chainA_plddt.npy"
        np.save(out_path, plddt_chain)

        plddt_stats[model_id] = {
            "mean": float(plddt_chain.mean()),
            "plddt_path": out_path,
        }

    return plddt_stats


def collect_boltz_chain_models(base_pred_dir: Path, target_chain="A"):
    """
    Select best Boltz model per prediction folder,
    but compute pLDDT ONLY for a given chain (default: A).

    Returns:
        dict compatible with plotting functions:
        {
            model_id: {
                "mean": float,
                "plddt_path": Path
            }
        }
    """

    plddt_stats = {}

    for pred_dir in base_pred_dir.iterdir():
        if not pred_dir.is_dir():
            continue

        best_score = -1
        best_plddt = None

        for npz_file in pred_dir.glob("plddt_*.npz"):

            data = np.load(npz_file)
            atom_plddt = data[data.files[0]]
            atom_plddt = np.asarray(atom_plddt)

            # Ensure atom-level shape
            atom_plddt = np.squeeze(atom_plddt)

            # Corresponding CIF file
            model_name = npz_file.stem.replace("plddt_", "")
            cif_path = pred_dir / f"{model_name}.cif"

            if not cif_path.exists():
                continue

            #  Core: extract ONLY chain A
            # --- reduce to residue-level ---
            atom_plddt = np.asarray(atom_plddt)
            while atom_plddt.ndim > 1:
                atom_plddt = atom_plddt.mean(axis=-1)

            # --- scale if needed ---
            if atom_plddt.max() <= 1.5:
                atom_plddt = atom_plddt * 100

            # --- get chain length (CDR = A) ---
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("model", str(cif_path))

            chain_length = 0
            for chain in structure.get_chains():
                if chain.id == "A":  # YAML bestätigt
                    chain_length = len([r for r in chain if r.id[0] == " "])
                    break

            # --- slice instead of mapping ---
            plddt_chain = atom_plddt[:chain_length]

            if len(plddt_chain) == 0:
                continue

            score = plddt_chain.mean()

            if score > best_score:
                best_score = score
                best_plddt = plddt_chain

        if best_plddt is None:
            continue

        model_id = pred_dir.name
        out_path = pred_dir / f"{model_id}_chainA_plddt.npy"

        np.save(out_path, best_plddt)

        plddt_stats[model_id] = {
            "mean": float(best_score),
            "plddt_path": out_path,
        }

    return plddt_stats


def extract_chain_residue_plddt_from_cif(cif_path, atom_plddt, target_chain="A"):
    """
    Extract residue-wise pLDDT for a specific chain.
    """

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", str(cif_path))

    residue_to_scores = defaultdict(list)

    atom_counter = 0

    for atom in structure.get_atoms():

        if atom.element == "H":
            continue

        residue = atom.get_parent()
        chain = residue.get_parent().id

        if residue.id[0] == "W":
            continue

        if atom_counter >= len(atom_plddt):
            break

        if chain != target_chain:
            atom_counter += 1
            continue

        res_key = residue.id[1]

        residue_to_scores[res_key].append(atom_plddt[atom_counter])

        atom_counter += 1

    plddt = np.array([
        np.mean(residue_to_scores[k])
        for k in sorted(residue_to_scores)
    ])

    return plddt


def collect_af_chain_models(base_pred_dir: Path, target_chain="A"):

    plddt_stats = {}

    for pred_dir in base_pred_dir.iterdir():

        if not pred_dir.is_dir():
            continue

        best_score = -1
        best_plddt = None

        for json_file in pred_dir.glob("*full_data_*.json"):

            with open(json_file) as fh:
                data = json.load(fh)

            atom_plddt = np.array(data["atom_plddts"])

            model_idx = json_file.stem.split("_")[-1]
            prefix = json_file.stem.rsplit("_full_data_", 1)[0]

            cif_path = pred_dir / f"{prefix}_model_{model_idx}.cif"

            if not cif_path.exists():
                continue

            plddt = extract_chain_residue_plddt_from_cif(
                cif_path,
                atom_plddt,
                target_chain,
            )

            score = plddt.mean()

            if score > best_score:
                best_score = score
                best_plddt = plddt

        if best_plddt is None:
            continue

        model_id = pred_dir.name
        out_path = pred_dir / f"{model_id}_chainA_plddt.npy"

        np.save(out_path, best_plddt)

        plddt_stats[model_id] = {
            "mean": float(best_score),
            "plddt_path": out_path,
        }

    return plddt_stats


def extract_chain_from_pdb(pdb_path, target_chain="A"):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", str(pdb_path))

    residues = []

    for model in structure:
        for chain in model:
            if chain.id != target_chain:
                continue

            for res in chain:
                if res.id[0] != " ":
                    continue
                residues.append(res)

    return len(residues)
