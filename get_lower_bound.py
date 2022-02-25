#!/usr/local/bin/python3
import argparse
from beam import BeamExperiment
from mixing_type import MixingType
from constants import *
from signal_processor import SignalProcessor

def is_mixing_too_small(hnl_mass, mixing_squared, num_samples, mixing_type):
    beam = BeamExperiment(beam_energy=400, nucleon_mass=1., \
        max_opening_angle=DETECTOR_OPENING_ANGLE, detector_length=DETECTOR_LENGTH, \
        detector_distance=DETECTOR_DISTANCE).with_mixing(mixing_squared)

    beam.start_dump(hnl_mass, num_samples=num_samples, mixing_type=mixing_type)
    mixing_too_small = SignalProcessor(beam).is_mixing_too_small()

    if mixing_too_small:
        print("Mixing too small")
    else:
        print("Mixing too large")
    return mixing_too_small

def main():
    hnl_masses = np.linspace(0.2, 1.2, 10)
    mixings_squared = np.linspace(10e-1, 1, 10)

    for mixing in mixings_squared:
        is_mixing_too_small(1.2, mixing, 10000, MixingType.electron)

if __name__ == "__main__":
    main()