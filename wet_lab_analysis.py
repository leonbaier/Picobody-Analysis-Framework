import re

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pypdf import PdfReader

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

        ax.legend()
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


def plot_bli_runs(bli_data: dict, date: str, run_names: list[str], save_path=None, title: str | None = None,
):

    if date not in bli_data:
        raise KeyError(f"Date '{date}' not found.")

    fig, ax = plt.subplots(figsize=(10, 6))

    for run_name in run_names:
        if run_name not in bli_data[date]["runs"]:
            raise KeyError(f"Run '{run_name}' not found for date '{date}'.")

        df = bli_data[date]["runs"][run_name]
        ax.plot(
            df["Time (s)"],
            df["Binding (nm)"],
            linewidth=2,
            label=run_name,)

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


def load_supr_dsf_export(csv_file: str | Path,
) -> dict[str, pd.DataFrame]:

    csv_file = Path(csv_file)

    with open(csv_file, "r", encoding="utf-8", errors="ignore",) as f:
        rows = [line.rstrip("\n").split(",") for line in f]

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
