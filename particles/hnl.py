
import numpy as np
from momentum4 import Momentum4
from utils import generate_samples, e_cos_theta_to_momentum4
from detector_signal import Signal
from .particle import Particle
from decay_type import DecayType
from mixing_type import MixingType
from constants import *

class HNL(Particle):
    def __init__(self, mass, beam=None, parent=None, momenta=[]):
        super().__init__(mass, beam, parent, momenta)
        self.signal = {}


    def __get_lepton_pair_lab_frame(self, lepton_samples: np.ndarray) -> Particle:
        # Sum of lepton pair momentum is anti-parallel to neutrino momentum
        di_lepton_rest_momenta = []
        for sample in lepton_samples:
            e_plus, e_minus = sample
            e_tot = e_plus + e_minus
            # Valid sample region is a triangle
            if e_tot <= self.m/2:
                continue
            e_nu = self.m - e_tot # Energy of the neutrino in rest frame
            cos_th = np.random.uniform()
            lepton_pair_inv_mass = np.sqrt(self.m**2 - 2*self.m*e_nu)
            di_lepton_rest_momenta.append(Momentum4.from_polar(e_tot, cos_th, 0, lepton_pair_inv_mass))
        # Cut the samples short since the sample region constraint will remove some points
        self.momenta = self.momenta[:len(di_lepton_rest_momenta)]
        lepton_pair = Particle(m=0, beam=self.beam, parent=self)
        lepton_pair.set_momenta(di_lepton_rest_momenta).boost(self.momenta)

        return lepton_pair
    
    def d_gamma_majorana_dEp_dEm(self, ep, em, decay_type = DecayType.CCNC):
        gr = SIN_WEINB**2 - 1/2
        gl = SIN_WEINB**2
        if decay_type == DecayType.CC:
            gr = 0
            gl = 0
        elif decay_type == DecayType.NC:
            gl = gl - 1
        output = ((1+gl)**2 + gr**2)*(self.m*(ep + em) - 2*(ep**2 + em**2))
        return output

    def d_gamma_dirac_dEp_dEm(self, ep, em, decay_type = DecayType.CCNC):
        gr = SIN_WEINB**2 - 1/2
        gl = SIN_WEINB**2
        if decay_type == DecayType.CC:
            gr = 0
            gl = 0
        elif decay_type == DecayType.NC:
            gl = gl - 1
        output = gr**2*em*(self.m - 2*em) + (1-gl)**2*ep*(self.m - 2*ep)
        return output

    def __partial_decay_rate_to_lepton_pair(self, mixing_type, decay_type=DecayType.CCNC):
        c1 = 0.25*(1 - 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        c2 = 0.25*(1 + 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        if mixing_type == MixingType.electron:
            # Ne -> e+ e- nu_e
            if decay_type == DecayType.CC:
                return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3))
            elif decay_type == DecayType.CCNC:
                return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3))*c2
            else:
                raise Exception("No value for only neutral current")
        elif mixing_type == MixingType.tau:
            # N_tau -> e+ e- nu_tau
            return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3))*c1

    def __add_linear_propagation_factors(self, length_detector, decay_rate):
        factors = []
        for p in self.momenta:
            factor = length_detector*self.m*decay_rate/p.get_total_momentum()
            factors.append(factor)
        self.propagation_factors = factors
        return self

    def __add_non_linear_propagation_factors(self, detector_length, detector_distance, partial_decay_rate, total_decay_rate):
        factors = []
        for p in self.momenta:
            ptot = p.get_total_momentum()
            factor1 = np.exp(-detector_distance*self.m/(ptot*total_decay_rate))
            factor2 = 1 - np.exp(-detector_length*self.m/(ptot*total_decay_rate))
            factor = factor1*factor2*(total_decay_rate/partial_decay_rate)
            factors.append(factor)
        self.propagation_factors = factors
        return self

    def __total_decay_rate(self):
        # TODO implement this
        pass

    def decay(self, num_samples, mixing_type: MixingType):
        # Decay N -> e+ e- v
        e_l_plus = np.linspace(0, self.m/2, 1000)
        e_l_minus = np.linspace(0, self.m/2, 1000)
        decay_type = DecayType.CC
        lepton_energy_samples = generate_samples(e_l_plus, e_l_minus, dist_func=lambda ep, em: self.d_gamma_dirac_dEp_dEm(ep, em, decay_type=decay_type), n_samples=num_samples)
        lepton_pair = self.__get_lepton_pair_lab_frame(lepton_energy_samples)

        if self.beam.linear_regime:
            self.__add_linear_propagation_factors(self.beam.DETECTOR_LENGTH, self.__partial_decay_rate_to_lepton_pair(mixing_type, decay_type=decay_type))
            print(f"Partial width: {self.__partial_decay_rate_to_lepton_pair(mixing_type, decay_type=decay_type)}")
        else:
            partial_decay = self.__partial_decay_rate_to_lepton_pair(mixing_type, decay_type=decay_type)
            total_decay = self.__total_decay_rate()
            self.__add_non_linear_propagation_factors(self.beam.DETECTOR_LENGTH, self.beam.DETECTOR_DISTANCE, partial_decay, total_decay)
        
        self.signal["e+e-v"] = [Signal(lepton_pair.momenta[i], self.propagation_factors[i]) for i in range(len(lepton_pair.momenta))]
        return self