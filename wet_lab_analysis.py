import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def load_akta_csv(csv_path: str | Path) -> dict:

    df = pd.read_csv(
        csv_path,
        sep="\t",
        header=None,
        dtype=str,
        engine="python",
        encoding="utf-16",
    )

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
                "labels": data["y"].astype(str).to_numpy(),
            }

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
            "y": data["y"].to_numpy(),
        }

    return result


def plot_affinity_chromatography_run(data: dict, run_name: str | list[str], signals: list[str],
                                     run_display_names: list[str] | None = None, save_path=None, title: str | None = None,
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

        run_colors = plt.cm.tab10.colors

        signal_styles = {
            "UV": "-",
            "Conductivity": "--",
            "Conc B": ":",
        }

        multiple_runs = len(run_name) > 1
        multiple_signals = len(signals) > 1

        if run_display_names is not None:
            if len(run_display_names) != len(run_name):
                raise ValueError(
                    "run_display_names must have same length as run_name."
                )

        for i, run in enumerate(run_name):

            if run not in data:
                raise KeyError(f"Run '{run}' not found.")

            run_data = data[run]

            display_run = (
                run_display_names[i]
                if run_display_names is not None
                else run
            )

            for signal in signals:

                if signal not in run_data:
                    raise KeyError(
                        f"Signal '{signal}' not found in run '{run}'."
                    )

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
                    linewidth=2.5,
                )

        first_run = run_name[0]
        first_signal = signals[0]

        ax.set_xlabel("Volume (ml)")

        ax.set_ylabel(
            f"{first_signal} "
            f"({data[first_run][first_signal]['y_label']})"
        )

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
        if signal not in run_data
    ]

    if invalid_signals:
        available = sorted(run_data.keys())

        raise ValueError(
            f"Signal(s) not found: {invalid_signals}\n"
            f"Available signals for '{run_name}':\n"
            f"{available}"
        )

    numeric_signals = [
        s for s in signals
        if run_data[s].get("type", "numeric") == "numeric"
    ]

    categorical_signals = [
        s for s in signals
        if run_data[s].get("type") == "categorical"
    ]

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
        "tab:brown",
    ]

    handles = []
    labels = []

    # ---------- numeric signals ----------
    for i, signal in enumerate(numeric_signals):
        signal_data = run_data[signal]

        ax = axes[min(i, len(axes) - 1)]

        signal_lower = signal.lower()

        is_uvvis = any(
            token in signal_lower
            for token in ["uv", "uv_vis", "uvvis"]
        )

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

        ax.set_ylabel(
            f"{signal} ({signal_data['y_label']})",
            color=color,
        )

        ax.tick_params(
            axis="y",
            labelcolor=color,
        )

        handles.append(line)
        labels.append(signal)

    # ---------- categorical signals ----------
    ymax = ax1.get_ylim()[1]

    for signal in categorical_signals:

        signal_data = run_data[signal]

        for x, label in zip(
                signal_data["x"],
                signal_data["labels"]
        ):
            ax1.axvline(
                x,
                color="grey",
                linestyle="--",
                linewidth=1.2,
                alpha=0.35,
                zorder=0,
            )

            ax1.text(
                x,
                ymax,
                str(label),
                rotation=90,
                fontsize=7,
                ha="center",
                va="top",
                color="grey",
                clip_on=True,
            )

    ax1.set_xlabel("Volume (ml)")

    if title is None:
        ax1.set_title(run_name)
    else:
        ax1.set_title(title)

    if handles:
        ax1.legend(handles, labels, loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Plot written to: {save_path}")
    else:
        plt.show()

    plt.close()