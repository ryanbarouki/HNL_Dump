import matplotlib.pyplot as plt
from beam import BeamExperiment
from mixing_type import MixingType
import numpy as np
import sys
import time
from hnl import HNL
import cross_sections as CS
import branching_ratios as BR
from constants import *

def apply_BEBC_cuts(hnls, e_min, pT_max) -> np.ndarray:
    """Assumes that samples are in the form (total Energy, total pT) of lepton pair"""
    cut_signal = []
    for hnl in hnls:
        if not hnl.signal:
            raise Exception("No signal object!")
        for signal in hnl.signal:
            energy = signal.momentum.get_energy()
            transverse_momentum = signal.momentum.get_transverse_momentum()
            if energy > e_min and transverse_momentum < pT_max:
                cut_signal.append(signal)
    return cut_signal

def get_total_normalised_hnl_decays(signal, num_samples, hnl_mass):
    fraction_decay = sum([signal.propagation_factor for signal in signal])/num_samples
    print(f"fraction decay: {fraction_decay}")
    flux_density = 32e-9 # electron neutrinos per proton per micro sr
    opening_angle = 46 # micro sr
    pot = 1.9e18 # protons on target
    electron_nu_massless_flux = flux_density*pot*opening_angle
    normalisation = electron_nu_massless_flux*(CS.P_TO_DPDM_X*BR.D_TO_E_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)
    normalisation_to_muon = electron_nu_massless_flux*(CS.P_TO_DPDM_X*BR.D_TO_MU_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_MU_NUMU_X + CS.P_TO_D0D0_X*BR.D0_TO_MU_NUMU_X)
    return fraction_decay*normalisation

def main():
    if len(sys.argv) < 3:
        raise Exception("Not enough arguments.\nPlease specify the HNL mass and the number of samples\n\
                        Usage: python BEBC_main.py <HNL mass in GeV> <number samples>")
    hnl_mass = float(sys.argv[1])
    num_samples = int(sys.argv[2]) 
    plot = False
    if len(sys.argv) > 3:
        plot = True

    beam = BeamExperiment(beam_energy=400, nucleon_mass=1.0, \
        max_opening_angle=DETECTOR_OPENING_ANGLE, detector_length=DETECTOR_LENGTH)
    
    start = time.time()

    beam.start_dump(hnl_mass, num_samples, MixingType.electron)
    hnls = beam.find_instances_of_type(HNL)
    cut_signal = apply_BEBC_cuts(hnls, e_min=0.8, pT_max=1.85)
    true_sample_number = num_samples * len(hnls)
    total_hnl_decays = get_total_normalised_hnl_decays(cut_signal, true_sample_number, hnl_mass)
    print(f"Mixing^2: {np.sqrt(2/total_hnl_decays)}")
   
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