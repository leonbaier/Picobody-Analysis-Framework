import re

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pypdf import PdfReader
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def create_fab_reports(sequence_file: str | Path, output_dir: str | Path,
) -> dict:

    sequence_file = Path(sequence_file)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True,)
    text = sequence_file.read_text(encoding="utf-8",)

    results = {}

    # --------------------------------------------------
    # Light chain
    # --------------------------------------------------

    lc_match = re.search(
        r"LC\s+SEQUENCE:\s*([A-Z*]+)\s+SIGNAL_PEPTIDE:\s*([A-Z]+)",
        text,
        re.DOTALL,
    )

    if lc_match is None:
        raise ValueError("Light chain block not found.")

    lc_sequence = lc_match.group(1)
    lc_signal_peptide = lc_match.group(2)

    # remove stop codon and everything after it
    lc_sequence = lc_sequence.split("*")[0]

    # remove signal peptide
    if lc_sequence.startswith(lc_signal_peptide):
        lc_sequence = lc_sequence[len(lc_signal_peptide):]

    # --------------------------------------------------
    # Variants
    # --------------------------------------------------

    variants = re.findall(
        (
            r"VARIANT:\s*([A-Za-z0-9_-]+)\s+"
            r"SEQUENCE:\s*([A-Z*]+)\s+"
            r"INSERT:\s*([A-Z]+)\s+"
            r"SIGNAL_PEPTIDE:\s*([A-Z]+)"
        ),
        text,
        re.DOTALL,
    )

    if len(variants) == 0:
        raise ValueError("No variants found.")

    for (
        variant_name,
        hc_sequence,
        insert_sequence,
        hc_signal_peptide,
    ) in variants:

        # ----------------------------------------------
        # sequence cleanup
        # ----------------------------------------------

        # (*marks stop)
        hc_sequence = hc_sequence.split("*")[0]

        if hc_sequence.startswith(hc_signal_peptide):
            hc_sequence = hc_sequence[len(hc_signal_peptide):]

        fab_sequence = (hc_sequence + lc_sequence)

        # ----------------------------------------------
        # ProtParam
        # ----------------------------------------------

        hc = ProteinAnalysis(hc_sequence,)
        lc = ProteinAnalysis(lc_sequence,)

        fab = ProteinAnalysis(fab_sequence,)

        hc_eps_red, hc_eps_ox = (hc.molar_extinction_coefficient())
        lc_eps_red, lc_eps_ox = (lc.molar_extinction_coefficient())
        fab_eps_red, fab_eps_ox = (fab.molar_extinction_coefficient())

        hc_mw = hc.molecular_weight()
        lc_mw = lc.molecular_weight()
        fab_mw = fab.molecular_weight()

        hc_abs_red = (hc_eps_red / hc_mw)
        hc_abs_ox = (hc_eps_ox / hc_mw)

        lc_abs_red = (lc_eps_red / lc_mw)
        lc_abs_ox = (lc_eps_ox / lc_mw)

        fab_abs_red = (fab_eps_red / fab_mw)
        fab_abs_ox = (fab_eps_ox / fab_mw)

        def get_residue_positions(sequence: str, residue: str,
        ) -> list:
            return[i + 1 for i, aa in enumerate(sequence) if aa == residue]

        result = {

            "HC": {
                "Length_aa":
                    len(hc_sequence),
                "MW_Da":
                    hc_mw,
                "pI":
                    hc.isoelectric_point(),
                "Epsilon_Oxidized":
                    hc_eps_ox,
                "Epsilon_Reduced":
                    hc_eps_red,
                "Abs0.1_Oxidized":
                    hc_abs_ox,
                "Abs0.1_Reduced":
                    hc_abs_red,
                "Trp":
                    hc_sequence.count("W"),
                "Tyr":
                    hc_sequence.count("Y"),
                "Cys":
                    hc_sequence.count("C"),
                "W_Positions":
                    get_residue_positions(hc_sequence, "W"),

                "Y_Positions":
                    get_residue_positions(hc_sequence, "Y"),

                "SUPR_DSF_Residues":
                    hc_sequence.count("W")
                    + hc_sequence.count("Y"),
            },

            "LC": {
                "Length_aa":
                    len(lc_sequence),
                "MW_Da":
                    lc_mw,
                "pI":
                    lc.isoelectric_point(),
                "Epsilon_Oxidized":
                    lc_eps_ox,
                "Epsilon_Reduced":
                    lc_eps_red,
                "Abs0.1_Oxidized":
                    lc_abs_ox,
                "Abs0.1_Reduced":
                    lc_abs_red,
                "Trp":
                    lc_sequence.count("W"),
                "Tyr":
                    lc_sequence.count("Y"),
                "Cys":
                    lc_sequence.count("C"),
                "W_Positions":
                    get_residue_positions(lc_sequence, "W"),

                "Y_Positions":
                    get_residue_positions(lc_sequence, "Y"),

                "SUPR_DSF_Residues":
                    lc_sequence.count("W")
                    + lc_sequence.count("Y"),
            },

            "Fab": {
                "Length_aa":
                    len(fab_sequence),
                "MW_Da":
                    fab_mw,
                "pI":
                    fab.isoelectric_point(),
                "Epsilon_Oxidized":
                    fab_eps_ox,
                "Epsilon_Reduced":
                    fab_eps_red,
                "Abs0.1_Oxidized":
                    fab_abs_ox,
                "Abs0.1_Reduced":
                    fab_abs_red,
                "Trp":
                    fab_sequence.count("W"),
                "Tyr":
                    fab_sequence.count("Y"),
                "Cys":
                    fab_sequence.count("C"),
                "W_Positions":
                    get_residue_positions(fab_sequence, "W"),

                "Y_Positions":
                    get_residue_positions(fab_sequence, "Y"),

                "SUPR_DSF_Residues":
                    fab_sequence.count("W")
                    + fab_sequence.count("Y"),
            },
            "Insert": {
                "Length_aa":
                    len(insert_sequence),

                "Trp":
                    insert_sequence.count("W"),

                "Tyr":
                    insert_sequence.count("Y"),

                "W_Positions":
                    get_residue_positions(insert_sequence, "W",),

                "Y_Positions":
                    get_residue_positions(insert_sequence, "Y",),

                "SUPR_DSF_Residues":
                    insert_sequence.count("W") + insert_sequence.count("Y"),
            }
        }

        results[variant_name] = result

        report = f"""
Variant: {variant_name}

Heavy Chain
-----------
Number of amino acids [-]: {result["HC"]["Length_aa"]}
Molecular weight {result["HC"]["MW_Da"]:.2f}
Theoretical pI [-]: {result["HC"]["pI"]:.2f}

Extinction coefficient oxidized [M^-1 cm^-1]: {result["HC"]["Epsilon_Oxidized"]}
Extinction coefficient reduced [M^-1 cm^-1]: {result["HC"]["Epsilon_Reduced"]}

Abs 0.1% oxidized [-]: {result["HC"]["Abs0.1_Oxidized"]:.3f}
Abs 0.1% reduced [-]: {result["HC"]["Abs0.1_Reduced"]:.3f}

Number of tryptophans [-]: {result["HC"]["Trp"]}
Number of tyrosines [-]: {result["HC"]["Tyr"]}
Number of cysteines [-]: {result["HC"]["Cys"]}


Light Chain
-----------
Number of amino acids [-]: {result["LC"]["Length_aa"]}
Molecular weight {result["LC"]["MW_Da"]:.2f}
Theoretical pI [-]: {result["LC"]["pI"]:.2f}

Extinction coefficient oxidized [M^-1 cm^-1]: {result["LC"]["Epsilon_Oxidized"]}
Extinction coefficient reduced [M^-1 cm^-1]: {result["LC"]["Epsilon_Reduced"]}

Abs 0.1% oxidized [-]: {result["LC"]["Abs0.1_Oxidized"]:.3f}
Abs 0.1% reduced [-]: {result["LC"]["Abs0.1_Reduced"]:.3f}

Number of tryptophans [-]: {result["LC"]["Trp"]}
Number of tyrosines [-]: {result["LC"]["Tyr"]}
Number of cysteines [-]: {result["LC"]["Cys"]}


Fab Fragment
------------
Number of amino acids [-]: {result["Fab"]["Length_aa"]}
Molecular weight {result["Fab"]["MW_Da"]:.2f}
Theoretical pI [-]: {result["Fab"]["pI"]:.2f}

Extinction coefficient oxidized [M^-1 cm^-1]: {result["Fab"]["Epsilon_Oxidized"]}
Extinction coefficient reduced [M^-1 cm^-1]: {result["Fab"]["Epsilon_Reduced"]}

Abs 0.1% oxidized [-]: {result["Fab"]["Abs0.1_Oxidized"]:.3f}
Abs 0.1% reduced [-]: {result["Fab"]["Abs0.1_Reduced"]:.3f}

Number of tryptophans [-]: {result["Fab"]["Trp"]}
Number of tyrosines [-]: {result["Fab"]["Tyr"]}
Number of cysteines [-]: {result["Fab"]["Cys"]}



SUPR-DSF Relevant Residues
--------------------------

Insert
------
Length [-]: {result["Insert"]["Length_aa"]}

Trp [-]: {result["Insert"]["Trp"]}
Tyr [-]: {result["Insert"]["Tyr"]}
Total fluorescent residues [-]: {result["Insert"]["SUPR_DSF_Residues"]}

Trp positions:
{result["Insert"]["W_Positions"]}

Tyr positions:
{result["Insert"]["Y_Positions"]}


Fab Fragment (HC + LC)
----------------------

Trp [-]: {result["Fab"]["Trp"]}
Tyr [-]: {result["Fab"]["Tyr"]}
Total fluorescent residues [-]: {result["Fab"]["SUPR_DSF_Residues"]}

Trp positions:
{result["Fab"]["W_Positions"]}

Tyr positions:
{result["Fab"]["Y_Positions"]}
""".strip()

        (output_dir / f"{variant_name}.txt").write_text(report, encoding="utf-8",)

        print(f"Created report: {variant_name}.txt")

    return results


def load_akta_csv(csv_path: str | Path) -> dict:

    encodings = [
        "utf-16",
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                csv_path,
                sep="\t",
                header=None,
                dtype=str,
                engine="python",
                encoding=encoding,)
            print(f"Loaded '{csv_path.name}' using encoding {encoding}")
            break

        except UnicodeError as e:
            last_error = e
    else:
        raise last_error

    signal_names = df.iloc[1]
    units = df.iloc[2]
    result = {}

    for col in range(0, len(df.columns) - 1, 2): # iteration in pairs (ml vs. X)
        signal_name = signal_names.iloc[col]

        if pd.isna(signal_name):
            continue

        signal_name = str(signal_name).strip()
        base_name = signal_name
        counter = 1

        # if signal_name is more than once in the list, add a counter
        while signal_name in result:
            counter += 1
            signal_name = f"{base_name}_{counter}"

        x_label = str(units.iloc[col]).strip()
        y_label = str(units.iloc[col + 1]).strip()

        data = df.iloc[3:, [col, col + 1]].copy()
        data.columns = ["x", "y"]

        # convert data to numeric
        data["x"] = pd.to_numeric(data["x"], errors="coerce")
        y_numeric = pd.to_numeric(data["y"], errors="coerce")

        # ---- categorical signal (Fraction, Event, Injection, ...) ----
        # in case of this condition, assume categorical
        if y_numeric.notna().sum() == 0:
            data = data.dropna(subset=["x"])

            if data.empty:
                continue

            result[signal_name] = {
                "type": "categorical",
                "x_label": x_label,
                "y_label": y_label,
                "x": data["x"].to_numpy(),
                "labels": data["y"].astype(str).to_numpy(),}
            continue

        # ---- numeric signal ----
        data["y"] = y_numeric
        data = data.dropna(subset=["x", "y"])

        if data.empty:
            continue

        result[signal_name] = {
            "type": "numeric",
            "x_label": x_label,
            "y_label": y_label,
            "x": data["x"].to_numpy(),
            "y": data["y"].to_numpy(),}

    return result


def plot_affinity_chromatography_run(data: dict, run_name: str | list[str], signals: list[str],
                                     run_display_names: list[str] | None = None, save_path=None, title: str | None = None,
                                     fraction_filter: list[tuple[str, str]] | None = None
):
    """
    Plot affinity chromatography data.csv (äkta output).

    Parameters
    ----------
    data : dict
        Dictionary returned by load_akta_csv().

    run_name : str | list[str]
        Single run name or a list of run names. If list then all variables of a single run ae plotted in the same color.
        Single run mode is more complex with many plotting possibilities like different colors for different variables
        and also the ability to plot categorical variables.

    signals : list[str]
        Signals to plot.

        Examples:
            ["UV"]
            ["UV", "Conductivity", "Conc B"]

        When multiple runs are supplied, only the requested
        signals are overlaid for all runs.

    run_display_names : list[str] | None
        Optional display names for runs when plotting multiple runs.
        Must have the same length as run_name if provided.

    save_path : Path | str | None
        Output path. If None, plot is displayed.

    title : str | None
        Optional custom plot title.

    Notes
    -----
    If run_name is a list, all requested runs are plotted
    in the same figure for direct comparison.
    """

    if isinstance(run_name, (list, tuple)):  # if list the only this block is performed otherwise skip

        fig, ax = plt.subplots(figsize=(12, 6))
        common_max_x = min(
            np.max(data[run][signals[0]]["x"])
            for run in run_name)

        run_colors = plt.cm.tab10.colors
        signal_styles = {
            "UV": "-",
            "Conductivity": "--",
            "Conc B": ":",}

        multiple_runs = len(run_name) > 1
        multiple_signals = len(signals) > 1

        if run_display_names is not None:
            if len(run_display_names) != len(run_name):
                raise ValueError("run_display_names must have same length as run_name.")

        for i, run in enumerate(run_name):

            if run not in data:
                raise KeyError(f"Run '{run}' not found.")

            run_data = data[run]
            display_run = (
                run_display_names[i]
                if run_display_names is not None
                else run)

            for signal in signals:
                if signal not in run_data:
                    raise KeyError(f"Signal '{signal}' not found in run '{run}'.")

                signal_data = run_data[signal]

                if signal_data.get("type") != "numeric":
                    continue

                linestyle = signal_styles.get(signal, "-")

                if multiple_runs:
                    if multiple_signals:
                        label = f"{display_run} - {signal}"
                    else:
                        label = display_run
                else:
                    label = signal

                ax.plot(
                    signal_data["x"],
                    signal_data["y"],
                    label=label,
                    color=run_colors[i % len(run_colors)],
                    linestyle=linestyle,
                    linewidth=2.5,)

        first_run = run_name[0]
        first_signal = signals[0]

        ax.set_xlabel("Volume (ml)")
        ax.set_xlim(0, common_max_x)

        ax.set_ylabel(
            f"{first_signal} "
            f"({data[first_run][first_signal]['y_label']})")

        ax.grid(alpha=0.3)

        if title is None:
            ax.set_title("Affinity Chromatography Comparison")
        else:
            ax.set_title(title)

        handles, labels = ax.get_legend_handles_labels()
        sort_order = []

        for label in labels:
            if label.endswith(" - UV"):
                sort_order.append(0)
            elif label.endswith(" - Conc B"):
                sort_order.append(1)
            else:
                sort_order.append(2)

        sorted_items = sorted(zip(sort_order, handles, labels), key=lambda x: x[0],)

        handles = [item[1] for item in sorted_items]
        labels = [item[2] for item in sorted_items]

        ax.legend(handles, labels,)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            print(f"Plot written to: {save_path}")
        else:
            plt.show()

        plt.close()

        return

    if run_name not in data:
        raise KeyError(f"Run '{run_name}' not found.")

    run_data = data[run_name]
    invalid_signals = [
        signal for signal in signals
        if signal not in run_data]

    if invalid_signals:
        available = sorted(run_data.keys())

        raise ValueError(
            f"Signal(s) not found: {invalid_signals}\n"
            f"Available signals for '{run_name}':\n"
            f"{available}")

    numeric_signals = [
        s for s in signals
        if run_data[s].get("type", "numeric") == "numeric"]

    categorical_signals = [
        s for s in signals
        if run_data[s].get("type") == "categorical"]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    axes = [ax1]

    if len(numeric_signals) >= 2:
        ax2 = ax1.twinx()
        axes.append(ax2)

    if len(numeric_signals) >= 3:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.10))
        axes.append(ax3)

    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",]

    handles = []
    labels = []

    # ---------- numeric signals ----------
    for i, signal in enumerate(numeric_signals):
        signal_data = run_data[signal]

        ax = axes[min(i, len(axes) - 1)]
        signal_lower = signal.lower()

        is_uvvis = any(
            token in signal_lower
            for token in ["uv", "uv_vis", "uvvis"])

        color = colors[i % len(colors)]

        line = ax.plot(
            signal_data["x"],
            signal_data["y"],
            color=color,
            linewidth=2.0 if is_uvvis else 1.5,
            alpha=1.0 if is_uvvis else 0.9,
            zorder=10 if is_uvvis else 1,
            label=signal,
        )[0]

        ax.set_ylabel(f"{signal} ({signal_data['y_label']})", color=color,)
        ax.tick_params(axis="y", labelcolor=color,)

        handles.append(line)
        labels.append(signal)

    # ---------- categorical signals ----------
    def fraction_in_ranges(fraction_label: str, ranges: list[tuple[str, str]],
    ) -> bool:
        for start, end in ranges:
            if start <= fraction_label <= end:
                return True
        return False

    for signal in categorical_signals:
        signal_data = run_data[signal]
        selected_x = []

        for x, label in zip(signal_data["x"], signal_data["labels"]):
            if fraction_filter is not None and not fraction_in_ranges(str(label), fraction_filter):
                continue
            selected_x.append(x)

        if selected_x:
            pool_patch = ax1.axvspan(
                min(selected_x),
                max(selected_x),
                color="gold",
                alpha=0.25,
                label="Collected fractions",)

    ax1.set_xlim(left=0)
    ax1.set_xlabel("Volume (ml)")

    if title is None:
        ax1.set_title(run_name)
    else:
        ax1.set_title(title)

    if selected_x:
        handles.append(pool_patch)
        labels.append("Collected fractions")
    if handles:
        ax1.legend(handles, labels, loc="upper right",)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Plot written to: {save_path}")
    else:
        plt.show()

    plt.close()


def load_sec_fraction_config(config_file):
    df = pd.read_csv(config_file, sep=";")

    selected_fractions = {}
    concentrations = {}

    for _, row in df.iterrows():

        run = row["run"]
        fraction = row["fraction"]
        concentration = row["concentration_mg_ml"]
        selected_fractions.setdefault(run, []).append(fraction)
        concentrations.setdefault(run, {})[fraction] = concentration

    return selected_fractions, concentrations


def report_sec_fractions(data: dict, config_file, fraction_signal: str = "Fraction", save_path=None,
):

    results = {}
    export_rows = []
    selected_fractions, concentrations = load_sec_fraction_config(config_file)

    for run_name, run_data in data.items():

        if fraction_signal not in run_data and not fraction_signal == "sec_fraction_config":
            print(f"\n{run_name}: no fraction signal found")
            continue

        frac_data = run_data[fraction_signal]
        x = frac_data["x"]
        labels = frac_data["labels"]

        run_fractions = (
            selected_fractions.get(run_name)
            if selected_fractions is not None
            else None)

        run_concentrations = (
            concentrations.get(run_name, {})
            if concentrations is not None
            else {})

        print("\n" + "=" * 100)
        print(run_name)
        print("=" * 100)

        total_protein_mg = 0
        run_results = {}

        for i in range(len(labels) - 1):
            current_fraction = str(labels[i]).strip()

            if current_fraction.lower() == "waste":
                continue

            if (
                run_fractions is not None
                and current_fraction not in run_fractions):
                continue

            volume_ml = float(x[i + 1] - x[i])

            row = {
                "run": run_name,
                "fraction": current_fraction,
                "volume_ml": volume_ml,}

            line = (
                f"{current_fraction:>10s} | "
                f"{volume_ml:6.2f} ml")

            if current_fraction in run_concentrations:

                conc = float(run_concentrations[current_fraction])

                protein_mg = conc * volume_ml
                total_protein_mg += protein_mg

                row["concentration_mg_ml"] = conc
                row["protein_mg"] = protein_mg
                line += f" | {conc:6.3f} mg/ml | {protein_mg:6.3f} mg"

            else:
                row["concentration_mg_ml"] = np.nan
                row["protein_mg"] = np.nan

            print(line)

            export_rows.append(row)
            run_results[current_fraction] = row

        print("-" * 100)
        print(
            f"Total protein amount: "
            f"{total_protein_mg:.3f} mg")

        export_rows.append({
            "run": run_name,
            "fraction": "TOTAL",
            "volume_ml": np.nan,
            "concentration_mg_ml": np.nan,
            "protein_mg": total_protein_mg,})
        results[run_name] = run_results

    if save_path is not None:
        pd.DataFrame(export_rows).to_csv(save_path, sep=";", index=False)
        print(f"\nFraction report written to:\n{save_path}")

    return results


def load_bli_dataset(folder: str | Path) -> dict:
    """
    Load all BLI csv files and automatically map run numbers
    to sample names using the corresponding BLItz PDF.

    Returns
    -------
    {
        "2026-07-27": {
            "mCloverV1_alone_0.200": dataframe,
            "mCloverV1_mClover_0.200": dataframe,
            ...
        }
    }
    """

    folder = Path(folder)
    data = {}

    for pdf_file in folder.glob("*.pdf"):
        date_match = re.search(r"(\d{8})", pdf_file.stem,)

        if not date_match:
            continue

        pdf_date = date_match.group(1)
        date_key = (
            f"{pdf_date[:4]}-"
            f"{pdf_date[4:6]}-"
            f"{pdf_date[6:8]}")

        # -----------------------------------------
        # extract run mapping from pdf
        # -----------------------------------------

        text = ""
        reader = PdfReader(pdf_file)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += "\n" + page_text

        mapping = {}

        step_matches = re.findall(r"(\d+)\s+([A-Za-z ]+?)\s+(\d+)\s+Tube", text,)
        matches = re.findall(r"\b(\d+)\s+1\s+1\s+0\s+([A-Za-z0-9_.\-]+)", text,)
        steps = [(step_name.strip(), int(duration),)
            for _, step_name, duration
            in step_matches]

        data.setdefault(date_key, {})
        data[date_key]["steps"] = steps
        data[date_key].setdefault("runs", {})

        for run_id, sample_name in matches:
            mapping[int(run_id)] = sample_name

        if not mapping:
            print(f"[BLI] No run mapping found in {pdf_file.name}")
            continue

        # -----------------------------------------
        # load matching csv
        # -----------------------------------------

        data.setdefault(date_key, {})
        csv_pattern = f"{date_key}_*.csv"

        for csv_file in folder.glob(csv_pattern):
            run_match = re.search(r"_(\d+)\.csv$", csv_file.name,)

            if not run_match:
                continue

            run_id = int(run_match.group(1))
            sample_name = mapping.get(run_id, f"run_{run_id}",)

            df = pd.read_csv(csv_file, names=["Time (s)", "Binding (nm)", "Step"], header=0,)
            df = df[["Time (s)", "Binding (nm)"]]
            df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors="coerce",)
            df["Binding (nm)"] = pd.to_numeric(df["Binding (nm)"], errors="coerce",)

            df = df.dropna()
            data[date_key]["runs"][sample_name] = df
    return data


def plot_bli_runs(bli_data: dict, date: str, run_names: list[str], run_display_names: list[str] | None = None,
                  save_path=None, title: str | None = None,
):

    if date not in bli_data:
        raise KeyError(f"Date '{date}' not found.")
    if run_display_names is not None:
        if len(run_display_names) != len(run_names):
            raise ValueError("run_display_names must have the same length as run_names.")

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, run_name in enumerate(run_names):
        if run_name not in bli_data[date]["runs"]:
            raise KeyError(f"Run '{run_name}' not found for date '{date}'.")

        label = (
            run_display_names[i]
            if run_display_names is not None
            else run_name)

        df = bli_data[date]["runs"][run_name]
        ax.plot(
            df["Time (s)"],
            df["Binding (nm)"],
            linewidth=2,
            label=label,)

    steps = bli_data[date]["steps"]
    cumulative_time = 0

    for step_name, duration in steps:
        ax.axvline(
            cumulative_time,
            color="red",
            linestyle="--",
            alpha=0.5,)
        x_center = (cumulative_time + duration / 2)

        ax.text(
            x_center,
            0.02,
            step_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="darkred",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=1.5,
            ),
        )

        cumulative_time += duration

    plt.xlabel("Time (s)")
    plt.ylabel("Binding (nm)")

    plt.xlim(0, cumulative_time)
    plt.ylim(-0.1, 1.0)

    if title is not None:
        plt.title(title)
    else:
        plt.title(f"BLI ({date})")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300,)
        print(f"Plot written to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_bli_phase_runs(bli_data: dict, runs: list[tuple[str, str]], phases: list[str],
                        run_display_names: list[str] | None = None, subtract_baseline: dict[str, str] | None = None,
                        save_path=None, title: str | None = None,
):
    if run_display_names is not None:
        if len(run_display_names) != len(runs):
            raise ValueError("run_display_names must have the same length as runs.")

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (date, run_name) in enumerate(runs):
        if date not in bli_data:
            raise KeyError(f"Date '{date}' not found.")
        if run_name not in bli_data[date]["runs"]:
            raise KeyError(f"Run '{run_name}' not found for date '{date}'.")

        label = (run_display_names[i] if run_display_names is not None else run_name)

        steps = bli_data[date]["steps"]
        phase_ranges = {}
        current_time = 0

        for step_name, duration in steps:
            phase_ranges[step_name] = (current_time, current_time + duration,)
            current_time += duration

        missing_phases = [phase for phase in phases if phase not in phase_ranges]

        if missing_phases:
            raise KeyError(f"Unknown phase(s): {missing_phases}")

        phase_intervals = [phase_ranges[phase] for phase in phases]

        df = bli_data[date]["runs"][run_name]
        mask = np.zeros(len(df), dtype=bool,)

        for start, end in phase_intervals:
            mask |= (
                    (df["Time (s)"] >= start)
                    &
                    (df["Time (s)"] <= end)
            )

        df_plot = df.loc[mask].copy()
        baseline_df = None

        if subtract_baseline is not None:
            if date not in subtract_baseline:
                raise KeyError(f"No baseline defined for {date}.")

            baseline_name = subtract_baseline[date]

            if (baseline_name not in bli_data[date]["runs"]):
                raise KeyError(f"Baseline run '{baseline_name}' not found for date '{date}'.")

            baseline_df = (bli_data[date]["runs"][baseline_name])

            baseline_values = np.interp(
                df_plot["Time (s)"],
                baseline_df["Time (s)"],
                baseline_df["Binding (nm)"],)
            df_plot["Binding (nm)"] -= (baseline_values)

        ax.plot(
            df_plot["Time (s)"],
            df_plot["Binding (nm)"],
            linewidth=2,
            label=label,)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Binding (nm)")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("BLI phase comparison")

    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300,)
        print(f"Plot written to: {save_path}")
    else:
        plt.show()

    plt.close()


def load_supr_dsf_export(csv_file: str | Path,
) -> dict[str, pd.DataFrame]:

    csv_file = Path(csv_file)

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    # suddenly the delimiter of the csv changed from the first to the second export
    delimiter = ";"
    if lines and lines[0].count(",") > lines[0].count(";"):
        delimiter = ","

    rows = [line.split(delimiter) for line in lines]

    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]

    raw = pd.DataFrame(rows)

    sample_info = raw.iloc[0]
    column_info = raw.iloc[1]

    data = {}
    col = 0

    while col + 5 < raw.shape[1]:

        well = sample_info.iloc[col]

        if not well:
            col += 1
            continue

        sample_name = sample_info.iloc[col + 1]
        cols = list(column_info.iloc[col:col + 6])

        block = raw.iloc[2:, col:col + 6,].copy()
        block.columns = cols
        block = block.apply(pd.to_numeric, errors="coerce",)
        block = block.dropna(subset=[cols[0]])

        data[f"{well}_{sample_name}"] = block
        col += 8

    return data


def load_all_supr_dsf_exports(save_dir: str | Path,
) -> dict:

    save_dir = Path(save_dir)
    all_data = {}

    for export_dir in save_dir.iterdir():

        if not export_dir.is_dir():
            continue

        thermal_file = (export_dir / "ThermalRamp_All.csv")

        if not thermal_file.exists():
            continue

        print(f"Loading {export_dir.name}")
        all_data[export_dir.name] = (load_supr_dsf_export( thermal_file))

    return all_data


def calculate_tonset_dbcm(fit_x: np.ndarray, fit_y: np.ndarray, peak_fraction: float = 0.05,
) -> float:

    baseline = np.mean(fit_y[:20])

    peak_height = (np.max(fit_y) - baseline)
    threshold = (baseline + peak_fraction * peak_height)

    tonset_idx = np.where(fit_y > threshold)[0][0]
    tonset = fit_x[tonset_idx]

    return tonset


# fit functions
def gaussian(x, amplitude, center, sigma,):
    return amplitude* np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def one_peak_model(x, offset, slope, a1, c1, s1,):
    return offset + slope * x + gaussian(x, a1, c1, s1,)

def two_peak_model(x, offset, slope, a1, c1, s1, a2, c2, s2,):
    return offset + slope * x + gaussian(x, a1, c1, s1,) + gaussian(x, a2, c2, s2,)


def plot_supr_dsf(supr_dsf_data: dict, experiment: str, sample: str, signal: str, smooth: bool = True,
        show_tm: bool = False, show_values: bool = False, save_path=None, title: str | None = None,
):

    if experiment not in supr_dsf_data:
        raise KeyError(f"Experiment '{experiment}' not found.")

    experiment_data = supr_dsf_data[experiment]

    sample_keys = [key
        for key in experiment_data
        if key.endswith(f"_{sample}")]

    if not sample_keys:
        raise KeyError(f"No samples found for '{sample}'.")

    plt.figure(figsize=(8, 5))

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",]

    for i, sample_key in enumerate(sample_keys):
        df = experiment_data[sample_key]
        plt.scatter(
            df["Temperature"],
            df[signal],
            s=12,
            color=colors[i % len(colors)],
            label=f"Replicate {i + 1}",)

    tm1 = None
    tm2 = None
    tonset = None

    if smooth and signal == "dBcm":

        reference_df = experiment_data[sample_keys[0]]
        x = reference_df["Temperature"].to_numpy()
        y_stack = np.vstack([
            experiment_data[key][signal].to_numpy()
            for key in sample_keys])

        y = np.mean(y_stack, axis=0,)
        peak_indices, _ = find_peaks( y, prominence=np.std(y) * 0.5,)
        peak_indices = peak_indices[np.argsort(y[peak_indices])[::-1]]

        n_transitions = min(len(peak_indices), 2,)

        if n_transitions == 0:
            peak = np.argmax(y)
            p0 = [np.min(y), 0, y[peak], x[peak], 2]

            params, _ = curve_fit(one_peak_model, x, y, p0=p0,
                bounds=(
                    [-np.inf, -np.inf, 0, min(x), 0],
                    [np.inf, np.inf, np.inf, max(x), 30],),
                maxfev=10000,)

            fit_y = one_peak_model(x, *params,)
            tonset = calculate_tonset_dbcm(
                fit_x=x,
                fit_y=fit_y,
                peak_fraction=0.05,
            )
            tm1 = params[3]

        elif n_transitions == 1:
            peak = peak_indices[0]
            p0 = [np.min(y), 0, y[peak], x[peak], 2]

            params, _ = curve_fit(one_peak_model, x, y, p0=p0,
                bounds=(
                    [-np.inf, -np.inf, 0, min(x), 0],
                    [np.inf, np.inf, np.inf, max(x), 30],),
                maxfev=10000,)

            fit_y = one_peak_model(x, *params,)
            tonset = calculate_tonset_dbcm(
                fit_x=x,
                fit_y=fit_y,
                peak_fraction=0.05,
            )
            tm1 = params[3]

        else:

            peak1 = peak_indices[0]
            peak2 = peak_indices[1]

            if x[peak1] > x[peak2]:
                peak1, peak2 = peak2, peak1

            p0 = [np.min(y), 0, y[peak1], x[peak1], 2, y[peak2], x[peak2], 2,]

            params, _ = curve_fit(two_peak_model, x, y, p0=p0,
                bounds=(
                    [-np.inf, -np.inf,
                     0, min(x), 0,
                     0, min(x), 0,],
                    [np.inf, np.inf,
                     np.inf, max(x), 30,
                     np.inf, max(x), 30,],
                ),
                maxfev=10000,)

            fit_y = two_peak_model(x, *params,)
            tonset = calculate_tonset_dbcm(
                fit_x=x,
                fit_y=fit_y,
                peak_fraction=0.05,
            )

            tm1 = params[3]
            tm2 = params[6]

        plt.plot(
            x,
            fit_y,
            color="black",
            linewidth=2.5,
            label="Peak fit",
            zorder=10,)

        if show_tm:
            plt.axvline(
                tm1,
                color="red",
                linestyle="--",
                linewidth=1.5,)

            if tm2 is not None:
                plt.axvline(
                    tm2,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,)

        if tonset is not None:
            plt.axvline(
                tonset,
                color="darkorange",
                linestyle="--",
                linewidth=1.5,
                label="Tonset",)

    if show_values:
        text_lines = []

        if tonset is not None:
            text_lines.append(f"Tonset = {tonset:.1f} °C")
        if tm1 is not None:
            text_lines.append(f"Tm1 = {tm1:.1f} °C")
        if tm2 is not None:
            text_lines.append(f"Tm2 = {tm2:.1f} °C")
        if text_lines:
            plt.gca().text(
                0.98,
                0.98,
                "\n".join(text_lines),
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9,),
            )

    plt.xlim(left=min(experiment_data[sample_keys[0]]["Temperature"]))
    plt.xlabel("Temperature (°C)")
    plt.ylabel(signal)

    if title:
        plt.title(title)
    else:
        plt.title(f"{sample} | {signal}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300,)
        print(f"Plot written to: {save_path}")
    else:
        plt.show()

    plt.close()
