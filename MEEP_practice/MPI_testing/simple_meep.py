import meep as mp
from meep import mpb

def main():
    # Define the simulation's geometry and parameters
    cell_size = mp.Vector3(16, 16, 0)
    geometry = [mp.Block(mp.Vector3(12, 1, mp.inf),
                         center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]

    # Create a simulation object
    resolution = 10
    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        resolution=resolution)

    # Define and run a simulation, here a simple transmission spectrum
    sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                         component=mp.Ez,
                         center=mp.Vector3(-7,0))]
    sim.sources = sources
    sim.run(until=200)

if __name__ == '__main__':
    main()
