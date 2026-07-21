from pathlib import Path
import shutil
import re

import MDAnalysis as mda
import matplotlib.pyplot as plt
import pandas as pd
from MDAnalysis.analysis import rms
from MDAnalysis.analysis import align
import numpy as np


def run_md_analysis(topology_file: Path, trajectory_file: Path, output_dir: Path, analyses: list, run_name: str,
):

    u = load_aligned_universe(
        topology_file=topology_file,
        trajectory_file=trajectory_file,)

    output_dir.mkdir(parents=True, exist_ok=True,)

    for analysis in analyses:
        analysis(u=u, output_dir=output_dir, run_name=run_name,)


def get_archived_md_runs(md_archive_root: Path,
) -> list[Path]:
    """
    Returns all archived MD run directories.

    Example:
        archive_root/
            2026-06-23_08-00-00/
                seq_1/
                seq_2/
            2026-06-24_09-00-00/
                seq_3/
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

    runs = []
    md_runs = get_archived_md_runs(archive_root)

    print(f"[MD Analysis] Found {len(md_runs)} archived runs.")

    for run_dir in md_runs:

        analysis_dir = run_dir / "analysis"

        analyses_current = list(md_analyses["default"]["analyses"])
        required_outputs_current = list(md_analyses["default"]["outputs"])

        if "with_ligand" in run_dir.name:
            analyses_current.extend(md_analyses["with_ligand"]["analyses"])
            required_outputs_current.extend(md_analyses["with_ligand"]["outputs"])

        if force_reanalysis:
            if analysis_dir.exists():
                shutil.rmtree(analysis_dir)

                print(f"[MD Analysis] Removed existing analysis: {run_dir.name}")

        analysis_dir.mkdir(parents=True, exist_ok=True,)

        if not force_reanalysis:
            already_done = all(
                (analysis_dir / output_file).exists()
                for output_file in required_outputs_current)

            if already_done:
                print(f"[MD Analysis] Skip {run_dir.name}")

                continue

        runs.append((
                analysis_dir,
                run_dir / "final.pdb",
                run_dir / "trajectory.dcd",
                analyses_current,
                run_dir.name,))

    return runs


def load_aligned_universe(topology_file: Path, trajectory_file: Path,
) -> mda.Universe:

    u = mda.Universe(topology_file, trajectory_file)
    align.AlignTraj(u, u, select="protein and name CA", in_memory=True).run()

    return u


def create_conditions_file(run_dir: Path, output_dir: Path, run_name: str,
):

    slurm_file = run_dir / "run_md.slurm"
    python_file = run_dir / "run_md.py"
    log_file = run_dir / "log.txt"
    topology_file = run_dir / "final.pdb"
    trajectory_file = run_dir / "trajectory.dcd"

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
    restraint_schedule = "Unknown"

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
{initial_k.group(1) if initial_k else "Unknown"}

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


def create_rmsd_analysis(u: mda.Universe, output_dir: Path, run_name: str
):

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