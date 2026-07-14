from openmm.app import *
from openmm import *
from openmm.unit import *

pdb = PDBFile(r"C:\Users\ge63laz\PycharmProjects\Masterthesis_Picobodies\pc_cluster_scripts/MolecularDynamics/seq_2_best_boltz_with_ligand/input.pdb")

# creates force field object
forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

# creates waterbox
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, padding=1.0*nanometer)
print("Atoms:", modeller.topology.getNumAtoms())


# create system from solvated model
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer)

# defines dynamics (T, friction, timestep) and device
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
platform = Platform.getPlatformByName('CUDA')

# creates simulation from structure, force field, dynamics, and device; sets start coordinates
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# minimize energy (if structure has bad geometry, steric conflicts,...)
print("Minimizing...")
simulation.minimizeEnergy()

# writes output to log.txt every steps, includes t, T, potential and total energy; writes trajectory
simulation.reporters.append(StateDataReporter(
    "log.txt",
    5000,
    step=True,
    time=True,
    temperature=True,
    potentialEnergy=True,
    totalEnergy=True))
simulation.reporters.append(DCDReporter("trajectory.dcd",10000))

# starts simulation (timesteps [2 fs] times steps is simulated time)
print("Starts simulation...")
simulation.step(5000)

positions = simulation.context.getState(getPositions=True).getPositions()

with open("final.pdb", "w") as f:
    PDBFile.writeFile(simulation.topology, positions, f)

print("Done")