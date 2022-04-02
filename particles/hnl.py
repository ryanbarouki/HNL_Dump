
import numpy as np
from particle_masses import *
from utils import DEBUG_AVERAGE_MOMENTUM
from .particle import Particle
from fundamental_constants import *
from logger import Logger
from hnl_decay_channels.e_pair import ElectronPair
from hnl_decay_channels.e_mu import ElectronMuon
from hnl_decay_channels.e_pi import ElectronPion
from hnl_decay_channels.mu_pair import MuonPair

class HNL(Particle):
    def __init__(self, mass, beam=None, parent=None, momenta=[], decay_mode=None):
        super().__init__(mass, beam, parent, momenta)
        self.decay_mode = decay_mode
        self.signal = {}
        self.average_propagation_factor = {}
        self.efficiency = {}

        # NOTE add more HNL decay channels here
        # TODO consider which decays are kinematically allowed
        self.decay_channels = {
            "e+pos+nu": ElectronPair,
            "mu+e+nu": ElectronMuon,
            "mu+mu+nu": MuonPair,
            "e+pi": ElectronPion
        }
        self.active_channels = []

    def __average_propagation_factor(self, length_detector, decay_rate):
        factors = []
        for p in self.momenta:
            factor = length_detector*self.m*decay_rate/p.get_total_momentum()
            factors.append(factor)
        return np.average(factors)

    def __avg_non_linear_propagation_factor(self, detector_length, detector_distance, partial_decay_rate, total_decay_rate):
        factors = []
        for p in self.momenta:
            ptot = p.get_total_momentum()
            factor1 = np.exp(-detector_distance*self.m*total_decay_rate/ptot)
            factor2 = 1 - np.exp(-detector_length*self.m*total_decay_rate/ptot)
            factor = factor1*factor2*(partial_decay_rate/total_decay_rate)
            factors.append(factor)
        return np.average(factors)

    def __total_decay_rate(self):
        total_decay_rate = 0
        for decay_channel in self.active_channels:
            total_decay_rate += decay_channel.partial_decay_rate()
        return total_decay_rate
        # c1 = 0.25*(1 - 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        # c2 = 0.25*(1 + 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        # if self.beam.mixing_type == MixingType.electron:
        #     return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3))*(c1 + c2 + 1) + self.__partial_decay_rate_to_electron_pi()
        # elif self.beam.mixing_type == MixingType.tau:
        #     return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3))*(2*c1)

    def __get_prop_factor_for_regime(self, partial_decay):
        if self.beam.linear_regime:
            return self.__average_propagation_factor(self.beam.detector_length, partial_decay)
        else:
            total_decay = self.__total_decay_rate()
            return self.__avg_non_linear_propagation_factor(self.beam.detector_length, self.beam.detector_distance, partial_decay, total_decay)

    def decay(self):
        self.acceptance = self.geometric_cut(0, self.beam.max_opening_angle)
        DEBUG_AVERAGE_MOMENTUM(self, "Average HNL momentum after angle cut")
        for channel_code in self.beam.channels:
            if channel_code in self.decay_channels:
                decay_channel = self.decay_channels[channel_code](beam=self.beam, parent=self)
                if not decay_channel.is_kinematically_allowed():
                    continue
                self.active_channels.append(decay_channel)
                partial_decay = decay_channel.partial_decay_rate()
                Logger().log(f"{channel_code} partial decay rate: {partial_decay}")
                self.average_propagation_factor[channel_code] = self.__get_prop_factor_for_regime(partial_decay)
                self.efficiency[channel_code] = decay_channel.decay()
                Logger().log(f"{channel_code} channel efficiency: {self.efficiency[channel_code]}")
        return self
    