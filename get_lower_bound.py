#!/usr/local/bin/python3
import argparse
from beam import BeamExperiment
from mixing_type import MixingType
from constants import *
from signal_processor import SignalProcessor
from logger import Logger

def total_decays_less_than_observed(hnl_mass, mixing_squared, num_samples, mixing_type):
    beam = BeamExperiment(beam_energy=400, nucleon_mass=1., \
        max_opening_angle=DETECTOR_OPENING_ANGLE, detector_length=DETECTOR_LENGTH, \
        detector_distance=DETECTOR_DISTANCE).with_mixing(mixing_squared)

    beam.start_dump(hnl_mass, num_samples=num_samples, mixing_type=mixing_type)
    total_decays_less_than_observed = SignalProcessor(beam).total_decays_less_than_observed()

    return total_decays_less_than_observed

def main():
    hnl_masses = np.linspace(0.2, 1.2, 10)
    mixings_squared = np.linspace(1e-3, 1e-2, 10)

    for mixing in mixings_squared:
        less_decays_than_observed = total_decays_less_than_observed(1.5, mixing, 10000, MixingType.electron)
        if less_decays_than_observed:
            print(f"Mixing: {mixing}, less decays than observed") 
        else:
            print(f"Mixing: {mixing}, more decays than observed") 
        print("================================================")


    # is_mixing_too_small(1.5, 6.21e-3, 10000, MixingType.electron)
if __name__ == "__main__":
    logger = Logger(debug=False)
    main()