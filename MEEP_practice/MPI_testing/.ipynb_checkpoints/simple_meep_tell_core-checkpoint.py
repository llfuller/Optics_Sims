# To run multi-core, use terminal command:
# mpirun -np 4 --map-by ppr:1:core python3 simple_meep_tell_core.py

import meep as mp
import os
import psutil
import subprocess


# def get_current_core():
#     command = "taskset -pc {}".format(os.getpid())
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     return result.stdout.split()[-1]

def get_current_core():
    with open(f"/proc/{os.getpid()}/stat") as f:
        fields = f.read().split()
    return fields[38]  # 39th field in 0-indexed list

def main():
    # Get the current process and the exact CPU core it's running on
    process = psutil.Process(os.getpid())
    process.cpu_affinity([rank])  # or any other logic to determine the right core

    core = process.cpu_affinity()[0]  # Get the first CPU in the affinity list
    # Print core information to a unique file for each process
    core_id = get_current_core()
    print(f"Current core ID: {core_id}")

    with open(f"process_{process.pid}_core_info.txt", "w") as f:
        f.write(f"Process {process.pid} is running on core: {core}\n {core_id}")

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

if __name__ == '__main__':
    main()
