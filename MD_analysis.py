from pathlib import Path
import shutil
import re

import MDAnalysis as mda
import matplotlib.pyplot as plt
import pandas as pd
from MDAnalysis.analysis import rms, align, distances
import numpy as np


def get_archived_md_runs(md_archive_root: Path,
) -> list[Path]:
    """
    Collects archived runs.
    """

    run_dirs = []

    for timestamp_dir in md_archive_root.iterdir():
        if not timestamp_dir.is_dir():
            continue
        for seq_dir in timestamp_dir.glob("seq_*"):
            if not seq_dir.is_dir():
                continue
            run_dirs.append(seq_dir)
    return sorted(run_dirs)


def get_md_analysis_runs(archive_root: Path, md_analyses: dict, force_reanalysis: bool = False,
) -> list[tuple[Path, Path, Path, list, str]]:
    """
        Identifies and validates archived MD runs that require analysis.

        Scans the archive directory, checks for the existence of necessary input
        files (topology, trajectory), and determines if the analysis has already
        been completed. Supports forcing a re-analysis which overwrites old data.

        Args:
            archive_root (Path): The root directory containing archived MD runs.
            md_analyses (dict): Configuration dictionary defining required outputs and analyses.
            force_reanalysis (bool): If True, existing analysis directories are deleted and recomputed. Default is False.

        Returns:
            list[tuple[Path, Path, Path, list, str]]: A list of tuples, each containing:
                - analysis_dir (Path): Target directory for outputs.
                - topology_file (Path): Path to the valid topology file.
                - trajectory_file (Path): Path to the valid trajectory file.
                - analyses_current (list): List of specific analysis functions to run.
                - run_name (str): Name of the specific MD run directory.
        """

    runs = []
    n_finished = 0
    n_empty = 0
    n_missing_topology = 0
    n_missing_traj = 0
    n_skipped = 0

    md_runs = get_archived_md_runs(archive_root)

    print(f"[MD Analysis] Found {len(md_runs)} archived runs.")

    for run_dir in md_runs:

        analysis_dir = run_dir / "analysis"
        trajectory_file = run_dir / "trajectory.dcd"

        analyses_current = list(md_analyses["default"]["analyses"])
        required_outputs_current = list(md_analyses["default"]["outputs"])

        if "with_ligand" in run_dir.name:
            analyses_current.extend(md_analyses["with_ligand"]["analyses"])
            required_outputs_current.extend(md_analyses["with_ligand"]["outputs"])

        if force_reanalysis:
            if analysis_dir.exists():
                shutil.rmtree(analysis_dir)
                print(f"[MD Analysis] Removed existing analysis: {run_dir.name}")

        if not force_reanalysis:
            already_done = all((analysis_dir / output_file).exists() for output_file in required_outputs_current)

            if already_done:
                n_skipped += 1
                print(f"[MD Analysis] Skip {run_dir.name}")
                continue

        topology_candidates = [
            run_dir / "final.pdb",
            run_dir / "initial.pdb",
            run_dir / "input.pdb",
            run_dir / "raw_input.pdb",]

        topology_file = next((p for p in topology_candidates if p.exists()), None,)

        if topology_file is None:
            n_missing_topology += 1
            print(f"[MD Analysis] No topology found: {run_dir}")
            continue
        if not trajectory_file.exists():
            print(f"[MD Analysis] No trajectory found: {run_dir}")
            continue
        if trajectory_file.stat().st_size == 0:
            n_empty += 1
            print(f"[MD Analysis] Empty trajectory: {run_dir}")
            continue

        try:
            test_u = mda.Universe(topology_file, trajectory_file,)
            len(test_u.trajectory)
        except Exception as e:
            print(f"[MD Analysis] Corrupt trajectory: {run_dir.name}")
            print(e)
            continue

        analysis_dir.mkdir(parents=True, exist_ok=True,)
        runs.append((
                analysis_dir,
                topology_file,
                trajectory_file,
                analyses_current,
                run_dir.name,))
        n_finished += 1

    print("\n[MD Analysis Summary]")
    print(f"Analysed: {n_finished}")
    print(f"Already done: {n_skipped}")
    print(f"Empty trajectories: {n_empty}")
    print(f"Missing topology: {n_missing_topology}")
    print(f"Missing trajectory: {n_missing_traj} \n")

    return runs


def load_aligned_universe(topology_file: Path, trajectory_file: Path,
) -> mda.Universe:
    """
        Loads an MD trajectory and aligns it to the initial topology.

        Uses MDAnalysis to load the topology and coordinate trajectory, and
        performs an in-memory structural alignment based on the C-alpha atoms
        to remove translational and rotational motions.

        Args:
            topology_file (Path): Path to the reference PDB topology file.
            trajectory_file (Path): Path to the DCD trajectory file.

        Returns:
            mda.Universe: The aligned MDAnalysis Universe object.
        """

    u = mda.Universe(topology_file, trajectory_file)
    align.AlignTraj(u, u, select="protein and name CA", in_memory=True).run()

    return u


def run_md_analysis(topology_file: Path, trajectory_file: Path, output_dir: Path, analyses: list, run_name: str,
):
    """
        Executes a series of analysis functions on a given MD simulation run.

        Loads the trajectory, aligns it to the initial topology in memory, and
        sequentially runs the provided analysis functions (e.g., RMSD, RMSF).

        Args:
            topology_file (Path): Path to the reference PDB topology file.
            trajectory_file (Path): Path to the DCD trajectory file.
            output_dir (Path): Directory where the analysis outputs will be saved.
            analyses (list): List of callable analysis functions to execute.
            run_name (str): Identifier of the simulation run.
        """
    u = load_aligned_universe(
        topology_file=topology_file,
        trajectory_file=trajectory_file,)
    output_dir.mkdir(parents=True, exist_ok=True,)

    for analysis in analyses:
        analysis(u=u, output_dir=output_dir, run_name=run_name,)


def create_rmsd_analysis(u: mda.Universe, output_dir: Path, run_name: str
):
    """
        Calculates the Root Mean Square Deviation (RMSD) of the trajectory.

        Computes the RMSD of the C-alpha atoms over the simulation time.
        The results are saved as a CSV file and plotted as a PNG image.

        Args:
            u (mda.Universe): The aligned MDAnalysis Universe.
            output_dir (Path): Directory where the CSV and PNG files will be saved.
            run_name (str): Identifier of the current run for plot titles.
        """
    R = rms.RMSD(u, u, select="protein and name CA",)
    R.run()

    rmsd_df = pd.DataFrame({
        "Frame": R.results.rmsd[:, 0],
        "Time_ps": R.results.rmsd[:, 1],
        "RMSD_Angstrom": R.results.rmsd[:, 2],})

    rmsd_df.to_csv(output_dir / "rmsd.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(rmsd_df["Time_ps"], rmsd_df["RMSD_Angstrom"],)
    plt.xlim(0, max(rmsd_df["Time_ps"]))

    plt.xlabel("Time (ps)")
    plt.ylabel("RMSD (Å)")
    plt.title(f"RMSD | {run_name.replace("_", " ")}")
    plt.tight_layout()

    plt.savefig(output_dir / "rmsd.png")
    plt.close()


def create_rmsf_analysis(u: mda.Universe, output_dir: Path, run_name: str
):
    """
        Calculates the Root Mean Square Fluctuation (RMSF) of the protein.

        Computes the per-residue RMSF for the C-alpha atoms across the trajectory
        to quantify local flexibility. The results are saved as a CSV file and
        plotted as a PNG image. Gaps in residue numbering are handled in the plot.

        Args:
            u (mda.Universe): The aligned MDAnalysis Universe.
            output_dir (Path): Directory where the CSV and PNG files will be saved.
            run_name (str): Identifier of the current run for plot titles.
        """

    protein = u.select_atoms("protein and name CA")

    rmsf_calc = rms.RMSF(protein).run()
    rmsf_df = pd.DataFrame({
        "Residue": protein.resids,
        "RMSF_Angstrom": rmsf_calc.results.rmsf,})

    rmsf_df.to_csv(output_dir / "rmsf.csv", index=False,)

    plt.figure(figsize=(8, 4))

    resids = protein.resids
    rmsf_values = rmsf_calc.results.rmsf

    start_idx = 0

    for i in range(1, len(resids)):
        if resids[i] != resids[i - 1] + 1:
            plt.plot(resids[start_idx:i], rmsf_values[start_idx:i],)

            start_idx = i

    plt.plot(resids[start_idx:], rmsf_values[start_idx:],)
    plt.xlim(0, max(resids))

    plt.xlabel("Residue")
    plt.ylabel("RMSF (Å)")
    plt.title(f"RMSF | {run_name.replace("_", " ")}")
    plt.tight_layout()

    plt.savefig(output_dir / "rmsf.png")
    plt.close()


def create_binder_target_distance_analysis(u: mda.Universe, output_dir: Path, run_name: str, binder_chain: str = "B",
                                           target_chain: str = "A",
):

    """
    Calculates the distance between the centers of mass of the binder and target.

    Iterates through the trajectory to compute the Euclidean distance between
    the center of mass of the binder chain and the target chain at each frame.

    Args:
        u (mda.Universe): The aligned MDAnalysis Universe.
        output_dir (Path): Directory for saving the CSV and PNG files.
        run_name (str): Identifier of the current run for plot titles.
        binder_chain (str): Segment ID of the binder (default: "B").
        target_chain (str): Segment ID of the target (default: "A").
    """

    binder = u.select_atoms(f"segid {binder_chain}")
    target = u.select_atoms(f"segid {target_chain}")

    if len(binder) == 0 or len(target) == 0:
        print("[Binding Analysis] Chains not found.")
        return

    times = []
    distances = []

    for ts in u.trajectory:
        distance = np.linalg.norm(binder.center_of_mass() - target.center_of_mass())

        times.append(ts.time)
        distances.append(distance)

    distance_df = pd.DataFrame({
        "Time_ps": times,
        "Distance_Angstrom": distances,})

    distance_df.to_csv(output_dir / "binder_target_distance.csv", index=False,)

    plt.figure(figsize=(8, 4))
    plt.plot(distance_df["Time_ps"], distance_df["Distance_Angstrom"],)
    plt.xlim(0, max(times))

    plt.xlabel("Time (ps)")
    plt.ylabel("Distance (Å)")
    plt.title(f"Binder-Target Distance | {run_name.replace("_", " ")}")
    plt.tight_layout()

    plt.savefig(output_dir / "binder_target_distance.png")
    plt.close()


def create_minimum_contact_distance_analysis(u: mda.Universe, output_dir: Path, run_name: str, binder_chain: str = "B",
        target_chain: str = "A",
):
    """
        Calculates the minimum atomic distance between the binder and the target.

        Computes a distance matrix between all atoms of the binder and the target
        for every frame and extracts the absolute minimum distance (closest contact).

        Args:
            u (mda.Universe): The aligned MDAnalysis Universe.
            output_dir (Path): Directory for saving the CSV and PNG files.
            run_name (str): Identifier of the current run for plot titles.
            binder_chain (str): Segment ID of the binder (default: "B").
            target_chain (str): Segment ID of the target (default: "A").
        """

    binder = u.select_atoms(f"segid {binder_chain}")
    target = u.select_atoms(f"segid {target_chain}")

    if len(binder) == 0 or len(target) == 0:
        print("[Minimum Distance] Chains not found.")
        return

    times = []
    min_distances = []

    for ts in u.trajectory:
        distance_matrix = distances.distance_array(binder.positions, target.positions,)
        min_distance = np.min(distance_matrix)

        times.append(ts.time)
        min_distances.append(min_distance)

    distance_df = pd.DataFrame({
        "Time_ps": times,
        "Min_Distance_Angstrom": min_distances,})

    distance_df.to_csv(output_dir / "minimum_contact_distance.csv", index=False,)

    plt.figure(figsize=(8, 4))
    plt.plot(distance_df["Time_ps"], distance_df["Min_Distance_Angstrom"],)

    plt.xlim(0, max(times),)
    plt.xlabel("Time (ps)")
    plt.ylabel("Minimum atom distance (Å)")
    plt.title(f"Minimum Contact Distance | {run_name.replace('_', ' ')}")
    plt.tight_layout()

    plt.savefig(output_dir / "minimum_contact_distance.png")

    plt.close()


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import cycle


def plot_combined_rmsd(run_paths_dict: dict, target_runs: list, tested_ids: list, comparison_ids: list,
                       save_path: Path, title: str = "Combined RMSD", rolling_window: int = 50):
    """
    Plots multiple RMSD trajectories in a single figure with color grouping.
    """
    plt.figure(figsize=(10, 6))

    # Linienstile für die Unterscheidbarkeit innerhalb der gleichen Farbgruppe
    tested_styles = cycle(['-', '--', '-.', ':'])
    comp_styles = cycle(['-', '--', '-.', ':'])

    for run_name in target_runs:
        if run_name in run_paths_dict and "rmsd_csv" in run_paths_dict[run_name]:
            csv_path = run_paths_dict[run_name]["rmsd_csv"]
            df = pd.read_csv(csv_path)

            label = run_name.replace("_best_af", "").replace("_best_boltz", "").replace("_with_ligand", "").replace(
                "_without_ligand", "").replace("_", " ")

            # Farb- und Stilzuweisung
            if any(run_name.startswith(f"{tid}_") for tid in tested_ids):
                color = "royalblue"
                ls = next(tested_styles)
            elif any(run_name.startswith(f"{cid}_") for cid in comparison_ids):
                color = "firebrick"
                ls = next(comp_styles)
            else:
                color = "gray"
                ls = "-"

            if rolling_window > 1:
                y_vals = df["RMSD_Angstrom"].rolling(window=rolling_window, min_periods=1).mean()
            else:
                y_vals = df["RMSD_Angstrom"]

            plt.plot(df["Time_ps"], y_vals, label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.85)

    plt.xlabel("Time (ps)")
    plt.ylabel("RMSD (Å)")
    plt.title(title)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_combined_rmsf(run_paths_dict: dict, target_runs: list, tested_ids: list, comparison_ids: list,
                       save_path: Path, title: str = "Combined RMSF", highlight_span: tuple = None):
    """
    Plots multiple RMSF profiles in a single figure with color grouping.
    """
    plt.figure(figsize=(10, 6))

    tested_styles = cycle(['-', '--', '-.', ':'])
    comp_styles = cycle(['-', '--', '-.', ':'])

    for run_name in target_runs:
        if run_name in run_paths_dict and "rmsf_csv" in run_paths_dict[run_name]:
            csv_path = run_paths_dict[run_name]["rmsf_csv"]
            df = pd.read_csv(csv_path)

            label = run_name.replace("_best_af", "").replace("_best_boltz", "").replace("_with_ligand", "").replace(
                "_without_ligand", "").replace("_", " ")

            # Farb- und Stilzuweisung
            if any(run_name.startswith(f"{tid}_") for tid in tested_ids):
                color = "royalblue"
                ls = next(tested_styles)
            elif any(run_name.startswith(f"{cid}_") for cid in comparison_ids):
                color = "firebrick"
                ls = next(comp_styles)
            else:
                color = "gray"
                ls = "-"

            resids = df["Residue"].values
            rmsf_values = df["RMSF_Angstrom"].values

            start_idx = 0
            for i in range(1, len(resids)):
                if resids[i] != resids[i - 1] + 1:
                    line, = plt.plot(resids[start_idx:i], rmsf_values[start_idx:i],
                                     color=color, linestyle=ls, linewidth=1.5, alpha=0.85)
                    if start_idx == 0:
                        line.set_label(label)
                    start_idx = i

            line, = plt.plot(resids[start_idx:], rmsf_values[start_idx:],
                             color=color, linestyle=ls, linewidth=1.5, alpha=0.85)
            if start_idx == 0:
                line.set_label(label)

    if highlight_span is not None:
        plt.axvspan(highlight_span[0], highlight_span[1], color='grey', alpha=0.2, label='Knob Region')

    plt.xlabel("Residue Index")
    plt.ylabel("RMSF (Å)")
    plt.title(title)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def create_conditions_file(run_dir: Path, output_dir: Path, run_name: str,
):
    """
        Generates a comprehensive metadata summary for an MD simulation run.

        Parses the OpenMM Python script, SLURM batch file, MD log file, and
        trajectory data using regular expressions to extract hardware settings,
        thermodynamic parameters, runtimes, and structural information.
        Writes a formatted report ('conditions.txt') to the analysis directory.

        Args:
            run_dir (Path): The original MD run directory containing scripts and logs.
            output_dir (Path): The analysis directory where 'conditions.txt' is saved.
            run_name (str): Identifier of the current run.
        """

    slurm_file = run_dir / "run_md.slurm"
    python_file = run_dir / "run_md.py"
    log_file = run_dir / "log.txt"
    trajectory_file = run_dir / "trajectory.dcd"

    topology_candidates = [
        run_dir / "final.pdb",
        run_dir / "initial.pdb",
        run_dir / "input.pdb",
        run_dir / "raw_input.pdb",]
    topology_file = next(
        (p for p in topology_candidates if p.exists()),
        None,)

    slurm_text = slurm_file.read_text(encoding="utf-8")
    python_text = python_file.read_text(encoding="utf-8")

    walltime = re.search(r"--time=([^\n]+)", slurm_text)
    cpus = re.search(r"--cpus-per-task=([^\n]+)", slurm_text)
    memory = re.search(r"--mem=([^\n]+)", slurm_text)
    node = re.search(r"--nodelist=([^\n]+)", slurm_text)
    conda_env = re.search(r"conda activate ([^\n]+)", slurm_text)

    forcefields = re.findall(r"ForceField\((.*?)\)", python_text, re.DOTALL,)
    platform = re.search(r"Platform\.getPlatformByName\('([^']+)'\)", python_text,)

    temperature = re.search(r"Langevin\w*Integrator\((.*?)\*kelvin", python_text,)
    friction = re.search(r"kelvin,\s*([0-9.]+)\s*/picosecond", python_text,)
    timestep = re.search(r"([0-9.]+)\*picoseconds", python_text,)
    all_steps = re.findall(r"simulation\.step\((\d+)\)", python_text,)
    cutoff = re.search(r"nonbondedCutoff=([0-9.]+)\*nanometer", python_text,)

    padding = re.search(r"padding=([0-9.]+)\*nanometer", python_text,)
    constraints = re.search(r"constraints=(\w+)", python_text,)

    nonbonded_method = re.search(r"nonbondedMethod=(\w+)", python_text,)

    log_interval = re.search(r'StateDataReporter\(\s*"log\.txt"\s*,\s*(\d+)', python_text, re.DOTALL,)
    dcd_interval = re.search(r'DCDReporter\("trajectory\.dcd"\s*,\s*(\d+)\)', python_text,)
    checkpoint_interval = re.search(r'CheckpointReporter\(\s*"checkpoint\.chk"\s*,\s*(\d+)\)', python_text,)

    water_model = re.search(r"'amber14/([^']+)'", python_text,)

    initial_k = re.search(r'addGlobalParameter\("k",\s*([0-9.]+)\)', python_text,)
    k_values = re.findall(r'setParameter\("k",\s*([0-9.]+)\)', python_text,)
    restraint_schedule = "None"

    if initial_k:
        schedule = [initial_k.group(1)] + k_values
        restraint_schedule = " -> ".join(schedule)

    simulation_time_ns = "Unknown"
    if timestep and all_steps:
        total_steps = sum(int(step) for step in all_steps)
        total_ps = (total_steps * float(timestep.group(1)))
        simulation_time_ns = (f"{total_ps / 1000:.1f}")

    runtime_s = "Unknown"
    runtime_h = "Unknown"
    gpu_name = "Unknown"
    run_status = "NOT_STARTED"

    mean_temperature = "Unknown"
    min_temperature = "Unknown"
    max_temperature = "Unknown"
    minimized_energy = "Unknown"

    if log_file.exists():
        log_text = log_file.read_text(encoding="utf-8", errors="ignore",)

        gpu = re.search(r"GPU:\s*(.+)", log_text,)
        walltime_s = re.search(r"Wall time \(s\):\s*([0-9.]+)", log_text,)
        walltime_h = re.search(r"Wall time \(h\):\s*([0-9.]+)", log_text,)
        energy_match = re.search(r"Minimized energy:\s*([^\n]+)", log_text,)

        if energy_match:
            minimized_energy = energy_match.group(1)

        if gpu:
            gpu_name = gpu.group(1)
        if walltime_s:
            runtime_s = walltime_s.group(1)
        if walltime_h:
            runtime_h = walltime_h.group(1)

        temperatures = []

        for line in log_text.splitlines():
            if line.startswith("#"):
                continue

            cols = line.split(",")
            if len(cols) >= 5:
                try:
                    temperatures.append(float(cols[4]))
                except ValueError:
                    pass

        if temperatures:
            mean_temperature = f"{sum(temperatures) / len(temperatures):.1f}"
            min_temperature = f"{min(temperatures):.1f}"
            max_temperature = f"{max(temperatures):.1f}"

    if (run_dir / "final.pdb").exists():
        run_status = "COMPLETED"
    elif trajectory_file.exists():
        try:
            u = mda.Universe(topology_file, trajectory_file)
            len(u.trajectory)
            run_status = "PARTIAL"
        except Exception:
            run_status = "CORRUPT"
    elif log_file.exists():
        log_text_lower = log_text.lower()
        if (
                "particle coordinate is nan" in log_text_lower
                or
                "openmmexception" in log_text_lower
                or
                "traceback" in log_text_lower):
            run_status = "FAILED"
        else:
            run_status = "INCOMPLETE"

    n_atoms = "Unknown"
    n_residues = "Unknown"
    n_frames = "Unknown"

    protein_atoms = "Unknown"
    protein_residues = "Unknown"
    n_waters = "Unknown"

    try:
        u = mda.Universe(topology_file, trajectory_file,)

        n_atoms = len(u.atoms)
        n_residues = len(u.residues)
        n_frames = len(u.trajectory)

        protein = u.select_atoms("protein")
        protein_atoms = len(protein)
        protein_residues = len(protein.residues)
        n_waters = len(u.select_atoms("resname HOH WAT").residues)
    except Exception:
        pass

    # big information document
    conditions = f"""Run name: {run_name}
Run status: {run_status}

System
------
Target included: {"Yes" if "with_ligand" in run_name else "No"}

Frames: {n_frames}

Total atoms: {n_atoms}
Total residues: {n_residues}

Protein atoms: {protein_atoms}
Protein residues: {protein_residues}

Water molecules: {n_waters}

SLURM
-----
Walltime limit: {walltime.group(1) if walltime else "Unknown"}
CPUs: {cpus.group(1) if cpus else "Unknown"}
GPU: {gpu_name}
Memory: {memory.group(1) if memory else "Unknown"}
Node: {node.group(1) if node else "Unknown"}
Conda environment: {conda_env.group(1) if conda_env else "Unknown"}

OpenMM
------
Platform: {platform.group(1) if platform else "Unknown"}

Force field(s):
{forcefields[0] if forcefields else "Unknown"}

Water model:
{water_model.group(1) if water_model else "Unknown"}

Water padding (nm):
{padding.group(1) if padding else "Unknown"}

Nonbonded method:
{nonbonded_method.group(1) if nonbonded_method else "Unknown"}

Nonbonded cutoff (nm):
{cutoff.group(1) if cutoff else "Unknown"}

Constraints:
{constraints.group(1) if constraints else "Unknown"}

Backbone restraints
-------------------
Initial k:
{initial_k.group(1) if initial_k else "None"}

Restraint schedule:
{restraint_schedule}

Simulation
----------
Temperature (K):
{temperature.group(1) if temperature else "Unknown"}

Friction (1/ps):
{friction.group(1) if friction else "Unknown"}

Timestep (ps):
{timestep.group(1) if timestep else "Unknown"}

Simulation time (ns):
{simulation_time_ns}

All simulation steps:
{", ".join(all_steps) if all_steps else "Unknown"}

Minimized energy (kJ/mol):
{minimized_energy}

Reporting
---------
Frames saved:
{n_frames}

log.txt interval (steps):
{log_interval.group(1) if log_interval else "Unknown"}

trajectory.dcd interval (steps):
{dcd_interval.group(1) if dcd_interval else "Unknown"}

checkpoint interval (steps):
{checkpoint_interval.group(1) if checkpoint_interval else "Unknown"}

Analysis
--------
RMSD: C-alpha aligned
RMSF: C-alpha
Binder-target distance: {"Yes" if "with_ligand" in run_name else "No"}

Runtime
-------
Wall time (s): {runtime_s}
Wall time (h): {runtime_h}

Temperature statistics
----------------------
Mean temperature (K): {mean_temperature}
Minimum temperature (K): {min_temperature}
Maximum temperature (K): {max_temperature}
"""

    (output_dir / "conditions.txt").write_text(conditions, encoding="utf-8",)
