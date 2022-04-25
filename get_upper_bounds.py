#!/usr/local/bin/python3
import numpy as np
import argparse
import matplotlib.pyplot as plt
from beam import BeamExperiment
from mixing_type import MixingType
import time
from signal_processor import SignalProcessor
from logger import Logger

def main():
    hnl_masses, mass_range, num_samples, debug, mixing_type, plot, save = parse_arguments()
    file = open(f"./upper_bound_data/upper_bounds_{mixing_type}.csv", "a")
    logger = Logger(debug=debug)
    upper_bounds = []

    if mass_range is not None:
        lower_mass, upper_mass, num_points = mass_range
        hnl_masses = np.linspace(lower_mass, upper_mass, int(num_points))

    for hnl_mass in hnl_masses:
        start = time.time()

        beam = BeamExperiment(mixing_type=mixing_type, num_samples=num_samples)

        beam.start_dump(hnl_mass)
        upper_bound_squared = SignalProcessor(beam).get_upper_bound()
        upper_bounds.append(upper_bound_squared)
        print(f"mass: {hnl_mass}, bound: {upper_bound_squared}")
        if save:
            file.write(f"{hnl_mass},{upper_bound_squared}\n")

        end = time.time()
        logger.log(f"Time taken: {end-start} seconds")

    if plot:
        plt.plot(hnl_masses, upper_bounds)
        plt.yscale('log')
        plt.show()

    file.close()

def plot_signal(cut_signal):
    energies = [momentum.get_energy() for momentum in cut_signal]
    pT = [momentum.get_transverse_momentum() for momentum in cut_signal]
    plt.hist2d(energies, pT, bins=100, range=[[0,250], [0, 2]])
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', type=float, nargs='+', help="Masses of the HNL in GeV")
    group.add_argument('--mass-range', type=float, nargs='+', help="Mass range '<lower> <higher> <number_of_points>' of the HNL in GeV")
    parser.add_argument('-n', type=int, help="Number of samples", required=True)
    parser.add_argument('--mixing', type=str, choices=['electron', 'tau'], help="valid values: 'electron' or 'tau'", required=True)
    parser.add_argument('--debug', action='store_true', help="Print debug logs")
    parser.add_argument('--plot', action='store_true', help="Plot upper bounds")
    parser.add_argument('--save', action='store_true', help="Save data to file")
    args = parser.parse_args()

    hnl_masses = args.m
    mass_range = args.mass_range
    num_samples = args.n
    debug = args.debug
    plot = args.plot
    save = args.save
    mixing_type = MixingType[args.mixing]
    return hnl_masses,mass_range,num_samples,debug,mixing_type,plot,save

if __name__ == "__main__":
    main()