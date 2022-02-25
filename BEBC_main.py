#!/usr/local/bin/python3
import argparse
import matplotlib.pyplot as plt
from beam import BeamExperiment
from mixing_type import MixingType
import time
from constants import *
from signal_processor import SignalProcessor

def main():
    hnl_mass, num_samples, plot, mixing_type = parse_arguments()

    start = time.time()

    beam = BeamExperiment(beam_energy=400, nucleon_mass=1., \
        max_opening_angle=DETECTOR_OPENING_ANGLE, detector_length=DETECTOR_LENGTH, \
        detector_distance=DETECTOR_DISTANCE)

    beam.start_dump(hnl_mass, num_samples=num_samples, mixing_type=mixing_type)
    upper_bound_squared, cut_signal = SignalProcessor(beam).get_upper_bound()
    print(f"U squared: {upper_bound_squared}")

    end = time.time()
    print(f"Time taken: {end-start} seconds")

    if plot:
        plot_signal(cut_signal)

def plot_signal(cut_signal):
    energies = [signal.momentum.get_energy() for signal in cut_signal]
    pT = [signal.momentum.get_transverse_momentum() for signal in cut_signal]
    plt.hist2d(energies, pT, bins=100, range=[[0,250], [0, 2]])
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=float, help="Mass of the HNL in GeV", required=True)
    parser.add_argument('-n', type=int, help="Number of samples", required=True)
    parser.add_argument('--mixing', type=str, choices=['electron', 'tau'], help="valid values: 'electron' or 'tau'", required=True)
    parser.add_argument('--plot', action='store_true', help="A boolean value whether to plot")
    args = parser.parse_args()

    hnl_mass = args.m
    num_samples = args.n
    plot = args.plot
    mixing_type = MixingType[args.mixing]
    return hnl_mass,num_samples,plot,mixing_type

if __name__ == "__main__":
    main()