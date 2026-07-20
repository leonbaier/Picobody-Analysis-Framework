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

    for col in range(0, len(df.columns) - 1, 2):

        signal_name = signal_names.iloc[col]

        if pd.isna(signal_name):
            continue

        signal_name = str(signal_name).strip()

        base_name = signal_name
        counter = 1

        while signal_name in result:
            counter += 1
            signal_name = f"{base_name}_{counter}"

        x_label = str(units.iloc[col]).strip()
        y_label = str(units.iloc[col + 1]).strip()

        data = df.iloc[3:, [col, col + 1]].copy()
        data.columns = ["x", "y"]

        data["x"] = pd.to_numeric(data["x"], errors="coerce")
        data["y"] = pd.to_numeric(data["y"], errors="coerce")

        data = data.dropna(subset=["x", "y"])

        if data.empty:
            continue

        result[signal_name] = {
            "x_label": x_label,
            "y_label": y_label,
            "x": data["x"].to_numpy(),
            "y": data["y"].to_numpy(),
        }

    return result


def plot_affinity_chromatography_run(data: dict, run_name: str, signals: list[str], save_path=None,
                                     title: str | None = None,
):

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

    fig, ax1 = plt.subplots(figsize=(12, 6))

    axes = [ax1]

    if len(signals) >= 2:
        ax2 = ax1.twinx()
        axes.append(ax2)

    if len(signals) >= 3:
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

    for i, signal in enumerate(signals):

        signal_data = run_data[signal]

        ax = axes[min(i, len(axes) - 1)]

        line = ax.plot(
            signal_data["x"],
            signal_data["y"],
            color=colors[i % len(colors)],
            linewidth=1.5,
            label=signal,
        )[0]

        ax.set_ylabel(
            f"{signal} ({signal_data['y_label']})",
            color=colors[i % len(colors)],
        )

        ax.tick_params(
            axis="y",
            labelcolor=colors[i % len(colors)],
        )

        handles.append(line)
        labels.append(signal)

    ax1.set_xlabel("Volume (ml)")

    if title is None:
        ax1.set_title(run_name)
    else:
        ax1.set_title(title)

    ax1.legend(handles, labels, loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Plot written to: {save_path}")
    else:
        plt.show()

    plt.close()