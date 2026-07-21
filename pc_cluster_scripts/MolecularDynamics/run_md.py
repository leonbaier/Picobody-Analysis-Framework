from openmm.app import *
from openmm import *
from openmm.unit import *
import time
import subprocess

gpu_name = "Unknown"
try:
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",],
        text=True,
    ).splitlines()[0]
except Exception:
    pass

pdb = PDBFile('input.pdb')

# creates force field object
forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

# creates waterbox
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, padding=1.2*nanometer)

# create system from solvated model
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0*nanometer,
    constraints=HBonds)

# defines dynamics (T, friction, timestep) and device
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
platform = Platform.getPlatformByName('CUDA')

# creates simulation from structure, force field, dynamics, and device; sets start coordinates
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)
print("Number of particles:", system.getNumParticles())

# track time
start_walltime = time.time()

# minimize energy (if structure has bad geometry, steric conflicts,...)
print("Minimizing...")
simulation.minimizeEnergy(maxIterations=10000)
state = simulation.context.getState(getEnergy=True)
print("Minimized energy:", state.getPotentialEnergy())
with open("log.txt", "w") as f:
    f.write(f"# Minimized energy: {state.getPotentialEnergy()}\n")

# store initial structure
state = simulation.context.getState(getPositions=True)
with open("initial.pdb", "w") as f:
    PDBFile.writeFile(
        simulation.topology,
        state.getPositions(),
        f)

# writes output to log.txt every steps, includes t, T, potential and total energy; writes trajectory
simulation.reporters.append(StateDataReporter(
    "log.txt",
    25000,
    step=True,
    time=True,
    temperature=True,
    potentialEnergy=True,
    totalEnergy=True))
simulation.reporters.append(DCDReporter("trajectory.dcd",50000))
simulation.reporters.append(CheckpointReporter( "checkpoint.chk", 10000000))

# starts simulation (timesteps [2 fs] times steps is simulated time)
print("Production run")
simulation.step(100000000)

positions = simulation.context.getState(getPositions=True).getPositions()

with open("final.pdb", "w") as f:
    PDBFile.writeFile(simulation.topology, positions, f)

elapsed = time.time() - start_walltime

with open("log.txt", "a") as f:
    f.write(f"\n# GPU: {gpu_name}\n")
    f.write(f"# Wall time (s): {elapsed:.2f}\n")
    f.write(f"# Wall time (h): {elapsed/3600:.2f}\n")

print("Done")
