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
modeller.addSolvent(forcefield, padding=1.0*nanometer)

# create system from solvated model
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0*nanometer,
    constraints=HBonds)

# Backbone restraints (pull stronger back if farther away)
restraint = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
restraint.addGlobalParameter("k", 1000.0) # initial k

restraint.addPerParticleParameter("x0")
restraint.addPerParticleParameter("y0")
restraint.addPerParticleParameter("z0")

for atom in modeller.topology.atoms():
    if atom.name in ["N", "CA", "C"]:
        pos = modeller.positions[atom.index]
        restraint.addParticle(atom.index,
            [pos.x.value_in_unit(nanometer), pos.y.value_in_unit(nanometer), pos.z.value_in_unit(nanometer)])
system.addForce(restraint)

# defines dynamics (T, friction, timestep) and device
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
platform = Platform.getPlatformByName('CUDA')

# creates simulation from structure, force field, dynamics, and device; sets start coordinates
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# track time
start_walltime = time.time()

# minimize energy (if structure has bad geometry, steric conflicts,...)
print("Minimizing...")
simulation.minimizeEnergy()

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
print("Starts simulation...")
print("Equilibration phase 1 (strong restraints)")
simulation.step(500000)  # 1 ns

print("Equilibration phase 2 (medium restraints)")
simulation.context.setParameter("k", 100.0)
simulation.step(500000)  # 1 ns

print("Equilibration phase 3 (weak restraints)")
simulation.context.setParameter("k", 10.0)
simulation.step(500000)  # 1 ns

print("Production run")
simulation.context.setParameter("k", 0.0)
simulation.step(98500000)

positions = simulation.context.getState(getPositions=True).getPositions()

with open("final.pdb", "w") as f:
    PDBFile.writeFile(simulation.topology, positions, f)

elapsed = time.time() - start_walltime

with open("log.txt", "a") as f:
    f.write(f"\n# GPU: {gpu_name}\n")
    f.write(f"# Wall time (s): {elapsed:.2f}\n")
    f.write(f"# Wall time (h): {elapsed/3600:.2f}\n")

print("Done")
