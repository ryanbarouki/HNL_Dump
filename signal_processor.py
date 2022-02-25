import numpy as np
from particles.hnl import HNL
import branching_ratios as BR
import cross_sections as CS

class SignalProcessor:
    def __init__(self, beam) -> None:
        self.beam = beam
    
    def get_upper_bound(self):
        hnls = self.beam.find_instances_of_type(HNL)
        channel = "e+e-v"
        if len(hnls) == 0:
            raise Exception("No HNLs!")
        efficiency, cut_signal = self.__apply_BEBC_cuts(hnls, channel=channel)
        prop_factor = 0
        acceptance = 0
        
        #TODO this is wrong, there should be some weighting
        for hnl in hnls:
            acceptance += hnl.acceptance
            prop_factor += hnl.average_propagation_factor 

        avg_prop_factor = prop_factor/len(hnls)
        acceptance = acceptance/len(hnls)
        print(f"avg propagation factor: {avg_prop_factor}")
        print(f"Acceptance: {acceptance}")
        total_flux = self.__get_normalised_hnl_flux_from_D_mesons(hnl_mass=hnls[0].m)
        total_decays = total_flux*avg_prop_factor*efficiency*acceptance
        upper_bound_squared = np.sqrt(3.5/total_decays)

        return upper_bound_squared, cut_signal

    def is_mixing_too_small(self):
        hnls = self.beam.find_instances_of_type(HNL)
        if len(hnls) == 0:
            raise Exception("No HNLs!")
        efficiency, cut_signal = self.__apply_BEBC_cuts(hnls, channel="e+e-v")
        hnl = hnls[0] #TODO: this won't work for multiple sources of HNL but we need the correct weighting

        avg_prop_factor = hnl.average_propagation_factor
        acceptance = hnl.acceptance
        normalisation = self.__get_normalised_hnl_flux_from_D_mesons(hnl_mass=hnl.m)
        observed_events = 3.5
        total_flux = normalisation*avg_prop_factor*efficiency*acceptance
        return total_flux < observed_events

    def __apply_BEBC_cuts(self, hnls, channel) -> np.ndarray:
        cut_signal = []
        total_signal = 0
        print(len(hnls))
        for hnl in hnls:
            if not hnl.signal:
                raise Exception("No signal object!")
            if channel == "e+e-v":
                e_min = 0.8 #GeV
                mT_max = 1.85 #GeV
                total_signal += len(hnl.signal[channel])
                for signal in hnl.signal[channel]:
                    energy = signal.momentum.get_energy()
                    transverse_mass = signal.momentum.get_transverse_mass()
                    if energy > e_min and transverse_mass < mT_max:
                        cut_signal.append(signal)
                print(f"Efficiency: {len(cut_signal)/len(hnl.signal[channel])}")
            else:
                raise Exception("Invalid channel or channel not implemented")
        return len(cut_signal) / total_signal, cut_signal

    def __get_normalised_hnl_flux_from_D_mesons(self, hnl_mass):
        electron_nu_massless_flux = 4.1e-4 * 2e18
        normalisation_D_mesons = electron_nu_massless_flux*(CS.P_TO_DPDM_X*BR.D_TO_E_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)
        # factor of 3 for the 3 decay channels (no need to treat them individually)
        return 3*normalisation_D_mesons