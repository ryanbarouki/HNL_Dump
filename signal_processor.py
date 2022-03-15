import numpy as np
from particles.tau_decay_modes import TauDecayModes
from mixing_type import MixingType
from particles.hnl import HNL
from particles.tau import Tau
from particles.DMeson import DMeson
from particles.DsMeson import DsMeson 
import branching_ratios as BR
import cross_sections as CS
from logger import Logger

ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2e18
OBSERVED_EVENTS = 3.5
class SignalProcessor:
    def __init__(self, beam) -> None:
        self.beam = beam
    
    def get_upper_bound(self):
        hnls = self.beam.find_instances_of_type(HNL)
        # TODO this channel string is everywhere - separately defined!
        channels = ["e+e-v"]#, "mu+e+nu"]
        if len(hnls) == 0:
            raise Exception("No HNLs!")

        total_decays = 0
        for channel in channels:
            total_decays_from_channel, cut_signal = self.get_total_decays(hnls, channel)
            total_decays += total_decays_from_channel

        upper_bound_squared = np.sqrt(OBSERVED_EVENTS/total_decays)
        return upper_bound_squared, cut_signal

    def get_total_decays(self, hnls, channel):
        logger = Logger()
        total_decays = 0
        for hnl in hnls:
            efficiency, cut_signal = self.__apply_BEBC_cuts(hnl, channel=channel, mixing_type=hnl.mixing_type)
            prop_factor = hnl.average_propagation_factor[channel]
            acceptance = hnl.acceptance
            total_flux = 0
            logger.log(f"Propagation factor: {prop_factor}")
            logger.log(f"Acceptance: {acceptance}")

            if hnl.mixing_type == MixingType.electron:
                if isinstance(hnl.parent, DMeson):
                    total_flux = self.__get_normalised_hnl_flux_from_DpDm_mesons(hnl_mass=hnl.m)
                    logger.log("Flux norm (D): {:e}".format(total_flux))
                elif isinstance(hnl.parent, DsMeson):
                    total_flux = self.__get_normalised_electron_hnl_flux_from_Ds_mesons(hnl_mass=hnl.m)
                    logger.log("Flux norm (Ds): {:e}".format(total_flux))
            elif hnl.mixing_type == MixingType.tau:
                if isinstance(hnl.parent, Tau):
                    if hnl.decay_mode == TauDecayModes.hnl_pi:
                        total_flux = self.__get_normalised_hnl_flux_from_tau_two_body(hnl_mass=hnl.m)
                        logger.log("Flux norm (tau 2-body): {:e}".format(total_flux))
                elif isinstance(hnl.parent, DsMeson):
                    total_flux = self.__get_normalised_tau_hnl_flux_from_Ds_mesons(hnl_mass=hnl.m)
                    logger.log("Flux norm (Ds): {:e}".format(total_flux))

            total_decays += total_flux*prop_factor*efficiency*acceptance
        return total_decays, cut_signal

    def total_decays_less_than_observed(self):
        hnls = self.beam.find_instances_of_type(HNL)
        if len(hnls) == 0:
            raise Exception("No HNLs!")
        total_decays, cut_signal = self.get_total_decays(hnls, channel="e+e-v")
        print(f"Total decays: {total_decays}")
        return total_decays < OBSERVED_EVENTS

    def __apply_BEBC_cuts(self, hnl, channel, mixing_type) -> np.ndarray:
        cut_signal = []
        total_signal = len(hnl.signal[channel])
        if not hnl.signal:
            raise Exception("No signal object!")
        
        e_min = 0
        mT_max = 0
        if mixing_type == MixingType.electron:
            e_min = 0.8 #GeV
            mT_max = 1.85 #GeV
        elif mixing_type == MixingType.tau:
            e_min = 0.8 #GeV
            mT_max = 0.19 #GeV
        # TODO what should the energy cut be for the HNL -> e mu nu channel?
        for momentum in hnl.signal[channel]:
            energy = momentum.get_energy()
            transverse_mass = momentum.get_transverse_mass()
            if energy > e_min and transverse_mass < mT_max:
                cut_signal.append(momentum)
        Logger().log(f"Efficiency: {len(cut_signal)/total_signal}")
        return len(cut_signal) / total_signal, cut_signal

    def __get_normalised_hnl_flux_from_DpDm_mesons(self, hnl_mass):
        normalisation_D_mesons = ELECTRON_NU_MASSLESS_FLUX*(CS.P_TO_DPDM_X*BR.D_TO_E_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)
        # factor of 3 for the 3 decay channels of HNLs (no need to treat them individually)
        return 3*normalisation_D_mesons

    def __get_normalised_tau_hnl_flux_from_Ds_mesons(self, hnl_mass):
        normalisation_D_mesons = ELECTRON_NU_MASSLESS_FLUX*(CS.P_TO_DSDS_X*BR.DS_TO_TAU_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)
        # factor of 3 for the 3 decay channels of HNLs (no need to treat them individually)
        return 3*normalisation_D_mesons

    def __get_normalised_electron_hnl_flux_from_Ds_mesons(self, hnl_mass):
        normalisation_D_mesons = ELECTRON_NU_MASSLESS_FLUX*(CS.P_TO_DSDS_X*BR.DS_TO_ELECTRON_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)
        # factor of 3 for the 3 decay channels of HNLs (no need to treat them individually)
        return 3*normalisation_D_mesons

    def __get_normalised_hnl_flux_from_tau_two_body(self, hnl_mass):
        normalisation_D_mesons = ELECTRON_NU_MASSLESS_FLUX*(CS.P_TO_DSDS_X*BR.DS_TO_TAU_X*BR.TAU_TO_PI_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)
        # factor of 3 for the 3 decay channels of HNLs (no need to treat them individually)
        return 3*normalisation_D_mesons