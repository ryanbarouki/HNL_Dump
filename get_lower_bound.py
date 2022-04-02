#!/usr/local/bin/python3
import argparse
import numpy as np
from beam import BeamExperiment
from mixing_type import MixingType
from signal_processor import SignalProcessor
from logger import Logger

def total_decays_less_than_observed(hnl_mass, mixing_squared, num_samples, mixing_type):
    beam = BeamExperiment(mixing_type=mixing_type, num_samples=num_samples).with_mixing(mixing_squared)

    beam.start_dump(hnl_mass)
    total_decays_less_than_observed = SignalProcessor(beam).total_decays_less_than_observed()

    return total_decays_less_than_observed

def main():
    lower_mixing = 1e-5
    upper_mixing = 1
    mixing = (upper_mixing + lower_mixing)/2
    prev_mixing = lower_mixing
    it = 0
    while abs((prev_mixing - mixing)/mixing) > 0.0005 and it < 25: 
        print(f"Mixing: {mixing}")
        less_decays_than_observed = total_decays_less_than_observed(1.5, mixing, 10000, MixingType.electron)
        if less_decays_than_observed:
            upper_mixing = mixing
        else:
            lower_mixing = mixing
        it += 1
        prev_mixing = mixing
        mixing = (upper_mixing + lower_mixing)/2
        print(abs(prev_mixing - mixing)/mixing)
    print("Lower bound: {:e}".format(mixing))    


if __name__ == "__main__":
    logger = Logger(debug=False)
    main()