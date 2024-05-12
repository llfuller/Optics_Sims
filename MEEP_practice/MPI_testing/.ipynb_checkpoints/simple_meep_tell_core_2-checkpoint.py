from mpi4py import MPI
import meep as mp
import os
import psutil
import subprocess


def set_cpu_affinity(available_cores):
    # Get the MPI communicator and the rank of this process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processes = comm.Get_size()
    
    # Determine which core to bind this process to
    # This example uses a simple modulus to distribute processes across available cores
    core_to_bind = available_cores[rank % len(available_cores)]

    # Set the CPU affinity for this process
    p = psutil.Process(os.getpid())
    p.cpu_affinity([core_to_bind])
    print(f"Process {rank} bound to core {core_to_bind}")

    print(f"Process {rank} bound to core {core_to_bind}")

    # Write core binding information to a file
    with open("core_binding_info.txt", "a") as file:
        file.write(f"Process {rank} finished on core {core_to_bind}\n")


# To run multi-core, use terminal command:
# mpirun -np 4 --map-by ppr:1:core python3 simple_meep_tell_core_2.py
# mpiexec -np 4 -bind-to core python3 simple_mpi_tell_core.py

if __name__ == "__main__":
    # List of available cores you want to use, adjust as necessary
    available_cores = [0, 1, 2, 3, 4, 5, 6, 7]

    # Set the CPU affinity based on rank and available cores
    set_cpu_affinity(available_cores)
    
    # Define the simulation's geometry and parameters
    cell_size = mp.Vector3(16, 16, 0)
    geometry = [mp.Block(mp.Vector3(12, 1, mp.inf),
                         center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]

    # Create a simulation object
    resolution = 70
    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        resolution=resolution)

    # Define and run a simulation, here a simple transmission spectrum
    sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                         component=mp.Ez,
                         center=mp.Vector3(-7,0))]

    mp.verbosity(3)
    
    sim.sources = sources
    sim.run(until=200)