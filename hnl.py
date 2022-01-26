
import imp
import numpy as np
from momentum4 import Momentum4
from utils import generate_samples, e_cos_theta_to_momentum4
from detector_signal import Signal
from particle import Particle
from neutrino import Neutrino
from decay_type import DecayType
from mixing_type import MixingType
from constants import *

class HNL(Particle):
    def three_body_tau_decay(self, e, cos_theta):
        aux0=3.+(((-4.*e)/self.parent.m)+(((0.5*((self.m**2)*(-4.+((6.*e)/self.parent.m))))/self.parent.m)/e))
        output=8.*((e**2)*((np.sqrt((1.-((e**-2.)*(self.m**2)))))*(aux0*(self.parent.m**-3.))))
        return output

    def __get_di_lepton_lab_frame(self, lepton_samples: np.ndarray):
        """Returns the total energy and total transverse momentum of lepton pair"""
        nu = Neutrino(self)
        # Sum of lepton pair momentum is anti-parallel to neutrino momentum
        nu_momenta = []
        for sample in lepton_samples:
            e_plus, e_minus = sample
            # Valid sample region is a triangle
            if e_plus + e_minus <= self.m/2:
                continue
            e_nu = self.m - (e_plus + e_minus) # Energy of the neutrino in rest frame
            cos_th = np.random.uniform()
            nu_momenta.append(Momentum4.from_polar(e_nu, cos_th, 0, nu.m))
        nu.set_momenta(nu_momenta)
        # Cut the samples short since the sample region constraint will remove some points
        self.momenta = self.momenta[:len(nu_momenta)]
        nu.set_momenta(nu_momenta).boost(self.momenta)
        nu_lab_energies = nu.get_energies()
        nu_lab_cos_theta = nu.get_cos_thetas()
        # hnl_decay_factors = hnl_lab_samples_cut[:,2]
        # p+ + p- has same momentum distribution as the neutrino in rest frame but energy is not the same
        # So we must get the energy of the lepton pair in the lab frame
        di_lepton_energies = self.get_energies() - nu_lab_energies
        di_lepton_trans_momentum = nu_lab_energies*np.sqrt(1-nu_lab_cos_theta**2)
        di_lepton_parallel_momentum = nu_lab_cos_theta*nu_lab_cos_theta

        di_lepton_momenta = []
        for i in range(len(di_lepton_energies)):
            di_lepton_momenta.append(Momentum4.from_e_pt_pp(di_lepton_energies[i], di_lepton_trans_momentum[i], di_lepton_parallel_momentum[i], 0, 0))

        return np.array(di_lepton_momenta)
    
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

    def __total_decay_rate_to_lepton_pair(self, mixing_type):
        # TODO this is only valid for electron mixing channel at the moment
        # TODO define weinberg angle in one place!!
        Gf = 1.166e-5 # GeV-2
        mN = self.m
        # xl = self.m / mN
        # log = np.log((1-3*xl**2 - (1-xl**2)*np.sqrt(1-4*xl**2))/((1 + np.sqrt(1-4*xl**2))*xl**2))
        c1 = 0.25*(1 - 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        # c2 = 0.5*SIN_WEINB**2*(2*SIN_WEINB**2 - 1)
        c3 = 0.25*(1 + 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        # c4 = 0.5*SIN_WEINB**2*(2*SIN_WEINB**2 + 1)
        # gamma = (Gf**2*mN**5/(192*np.pi**3))*((c1*(1-delta) + c3*delta)*((1-14*xl**2 - 2*xl**4 - 12*xl**6)*np.sqrt(1-4*xl**2) / 
        #         + 12*xl**4*(xl**4 - 1)*log) + 4*(c2*(1-delta)+c4*delta)*(xl**2*(2 + 10*xl**2 - 12*xl**4)*np.sqrt(1-4*xl**2) /
        #         + 6*xl**4*(1 - 2*xl**2 + 2*xl**4)*log))
        if mixing_type == MixingType.electron:
            return (Gf**2*mN**5/(192*np.pi**3))*(1+c3+c1)
        else:
            return Exception("Muon and Tau not implemented")

    def __add_propagation_factors(self, length_detector, decay_rate):
        # TODO Currently only in linear regime
        factors = []
        for p in self.momenta:
            factor = length_detector*self.m*decay_rate/p.get_total_momentum()
            factors.append(factor)
        self.propagation_factors = factors
        return self

    def decay(self, num_samples, mixing_type: MixingType):
        e_l_plus = np.linspace(0, self.m/2, 1000)
        e_l_minus = np.linspace(0, self.m/2, 1000)
        lepton_energy_samples = generate_samples(e_l_plus, e_l_minus, dist_func=lambda ep, em: self.d_gamma_majorana_dEp_dEm(ep, em, decay_type=DecayType.CCNC), n_samples=num_samples)
        di_lepton_momenta = self.__get_di_lepton_lab_frame(lepton_energy_samples)
        self.__add_propagation_factors(self.beam.DETECTOR_LENGTH, self.__total_decay_rate_to_lepton_pair(mixing_type))
        self.signal = [Signal(di_lepton_momenta[i], self.propagation_factors[i]) for i in range(len(di_lepton_momenta))]
        return self