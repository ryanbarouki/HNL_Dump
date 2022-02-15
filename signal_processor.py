import numpy as np
from particles.hnl import HNL
import branching_ratios as BR
import cross_sections as CS

class SignalProcessor:
    def __init__(self, beam) -> None:
        self.beam = beam
    
    def get_bound(self):
        hnls = self.beam.find_instances_of_type(HNL)
        channel = "e+e-v"
        if len(hnls) == 0:
            raise Exception("No HNLs!")
        efficiency, cut_signal = self.__apply_BEBC_cuts(hnls, channel=channel)
        prop_factor = 0
        acceptance = 0
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

        self.__debug_print_average_p(hnls)
        return upper_bound_squared, cut_signal

    def is_mixing_too_small(self):
        hnls = self.beam.find_instances_of_type(HNL)
        if len(hnls) == 0:
            raise Exception("No HNLs!")
        cut_signal = self.__apply_BEBC_cuts(hnls, channel="e+e-v")
        normalised_hnl_decays = self.__get_normalised_hnl_flux_from_D_mesons(cut_signal, hnl_mass=hnls[0].m)
        observed_events = 2
        return normalised_hnl_decays < observed_events

    def __debug_print_average_p(self, hnls):
        total_p = []
        for hnl in hnls:
            for p in hnl.momenta:
                total_p.append(p.get_total_momentum())
        print(f"average p: {sum(total_p)/len(total_p)}")

    def __apply_BEBC_cuts(self, hnls, channel) -> np.ndarray:
        cut_signal = []
        total_signal = 0
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
        # factor of 2 to include anti-neutrinos
        flux_density = 2*32e-9 # electron neutrinos per proton per micro sr
        opening_angle = 4*np.pi*10**6#46 # micro sr
        pot = 2.2e18 # protons on target
        electron_nu_massless_flux = flux_density*pot*opening_angle
        normalisation_D_mesons = electron_nu_massless_flux*(CS.P_TO_DPDM_X*BR.D_TO_E_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)
        # factor of 4 for the 4 decay channels (no need to treat them individually)
        return 4*normalisation_D_mesons