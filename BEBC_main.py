#!/usr/local/bin/python3
import matplotlib.pyplot as plt
from beam import BeamExperiment
from mixing_type import MixingType
import sys
import time
from constants import *
from signal_processor import SignalProcessor

def main():
    if len(sys.argv) < 4:
        raise Exception("Not enough arguments.\nPlease specify the HNL mass, the number of samples and the mixing type\n\
                        Usage: python BEBC_main.py <HNL mass in GeV> <number samples> -e/-t (-p)")
    hnl_mass = float(sys.argv[1])
    num_samples = int(sys.argv[2]) 
    plot = False
    mixing_type = MixingType.tau
    for flag in sys.argv[3:]:
        if flag == "-e":
            mixing_type = MixingType.electron
        elif flag == "-t":
            mixing_type == MixingType.tau

    if "-p" in sys.argv:
        plot = True

    start = time.time()

    beam = BeamExperiment(beam_energy=400, nucleon_mass=58.0, \
        max_opening_angle=DETECTOR_OPENING_ANGLE, detector_length=DETECTOR_LENGTH, \
        detector_distance=DETECTOR_DISTANCE)

    beam.start_dump(hnl_mass, num_samples=num_samples, mixing_type=mixing_type)
    upper_bound_squared, cut_signal = SignalProcessor(beam).get_bound()
    print(f"U squared: {upper_bound_squared}")

    end = time.time()
    print(f"Time taken: {end-start} seconds")

    # plot lepton pair histogram
    if plot:
        energies = [signal.momentum.get_energy() for signal in cut_signal]
        pT = [signal.momentum.get_transverse_momentum() for signal in cut_signal]
        plt.hist2d(energies, pT, bins=100, range=[[0,250], [0, 2]])
        plt.show()


if __name__ == "__main__":
    main()