
import matplotlib.pyplot as plt
from beam import BeamExperiment
from mixing_type import MixingType
import time
from signal_processor import SignalProcessor
from logger import Logger
import numpy as np

def main():
    logger = Logger(debug=False)
    upper_bounds = []
    hnl_masses = np.linspace(0.2, 1.65, 36, endpoint=False)
    for hnl_mass in hnl_masses:
        start = time.time()
        beam = BeamExperiment(mixing_type=MixingType.tau, num_samples=10000)
        beam.start_dump(hnl_mass)
        upper_bound_squared = SignalProcessor(beam).get_upper_bound()
        upper_bounds.append(upper_bound_squared)
        print(f"mass: {hnl_mass}, bound: {upper_bound_squared}")
        end = time.time()
        logger.log(f"Time taken: {end-start} seconds")
    
    plt.plot(hnl_masses, upper_bounds)
    plt.yscale('log')
    plt.show()
    
if __name__ == "__main__":
    main()
