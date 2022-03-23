
import numpy as np
from particles.electron import Electron
from particles.muon import Muon
from particles.pion import Pion
from particle_masses import *
from momentum4 import Momentum4
from utils import generate_samples, get_two_body_momenta, DEBUG_AVERAGE_MOMENTUM
from .particle import Particle
from decay_type import DecayType
from mixing_type import MixingType
from constants import *
from logger import Logger

class HNL(Particle):
    def __init__(self, mass, mixing_type, beam=None, parent=None, momenta=[], decay_mode=None):
        super().__init__(mass, beam, parent, momenta)
        self.mixing_type = mixing_type
        self.decay_mode = decay_mode
        self.signal = {}
        self.average_propagation_factor = {}
        self.efficiency = {}
        self.decay_funcs = {
            "e+pos+nu": self.decay_to_e_pair,
            "mu+e+nu": self.decay_to_e_mu,
            "e+pi": self.decay_to_e_pi
        }


    def __get_lepton_pair_lab_frame(self, lepton_samples: np.ndarray) -> Particle:
        # Sum of lepton pair momentum is anti-parallel to neutrino momentum
        di_lepton_rest_momenta = []
        for sample in lepton_samples:
            e1, e2 = sample
            e_tot = e1 + e2
            e3 = self.m - e_tot # Energy of the neutrino in rest frame of HNL
            cos_th = np.random.uniform()
            lepton_pair_inv_mass = np.sqrt(self.m**2 - 2*self.m*e3)
            di_lepton_rest_momenta.append(Momentum4.from_polar(e_tot, cos_th, 0, lepton_pair_inv_mass))
        lepton_pair = Particle(m=0, beam=self.beam, parent=self)
        lepton_pair.set_momenta(di_lepton_rest_momenta).boost(self.momenta)
        return lepton_pair
    
    def __electron_positron_dist_majorana(self, ep, em, decay_type = DecayType.CCNC):
        gr = SIN_WEINB**2 - 1/2
        gl = SIN_WEINB**2
        if decay_type == DecayType.CC:
            gr = 0
            gl = 0
        elif decay_type == DecayType.NC:
            gl = gl - 1
        output = ((1+gl)**2 + gr**2)*(self.m*(ep + em) - 2*(ep**2 + em**2))
        return output

    def __electron_positron_dist_dirac(self, ep, em, decay_type=DecayType.CCNC):
        gr = SIN_WEINB**2 - 1/2
        gl = SIN_WEINB**2
        if decay_type == DecayType.CC:
            gr = 0
            gl = 0
        elif decay_type == DecayType.NC:
            gl = gl - 1
        output = gr**2*em*(self.m - 2*em) + (1-gl)**2*ep*(self.m - 2*ep)
        return output

    def __electron_muon_dist(self, e_elec, e_muon):
        # https://arxiv.org/abs/hep-ph/9703333
        xm = MUON_MASS/self.m
        x_mu = 2*e_muon/self.m
        return x_mu*(1 - x_mu + xm**2)

    def __partial_decay_rate_to_electron_pair(self, mixing_type, decay_type=DecayType.CCNC):
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

    def __partial_decay_rate_to_electron_muon(self, mixing_type, decay_type=DecayType.CCNC):
        if mixing_type == MixingType.electron:
            # https://arxiv.org/abs/hep-ph/9703333
            xm = MUON_MASS/self.m
            func_form = 1 - 8*xm**2 + 8*xm**6 + xm**8 - 12*xm**4*np.log(xm**2)
            return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3)*func_form)

    def __partial_decay_rate_to_electron_pi(self, mixing_type):
        if mixing_type == MixingType.electron:
            phase_factor = 1 # TODO find this factor https://arxiv.org/pdf/1805.08567.pdf
            return self.beam.mixing_squared*(GF**2*self.m**3*F_PI**2/(16*np.pi))*phase_factor

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

    def __total_decay_rate(self, mixing_type):
        c1 = 0.25*(1 - 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        c2 = 0.25*(1 + 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        if mixing_type == MixingType.electron:
            return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3))*(c1 + c2 + 1)
        elif mixing_type == MixingType.tau:
            # TODO double check this 
            return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3))*(2*c1)

    def __get_prop_factor_for_regime(self, mixing_type, partial_decay):
        if self.beam.linear_regime:
            return self.__average_propagation_factor(self.beam.DETECTOR_LENGTH, partial_decay)
        else:
            total_decay = self.__total_decay_rate(mixing_type)
            return self.__avg_non_linear_propagation_factor(self.beam.DETECTOR_LENGTH, self.beam.DETECTOR_DISTANCE, partial_decay, total_decay)

    def decay(self, num_samples, mixing_type: MixingType):
        self.acceptance = self.geometric_cut(0, self.beam.MAX_OPENING_ANGLE)
        DEBUG_AVERAGE_MOMENTUM(self, "Average HNL momentum after angle cut")
        if mixing_type == MixingType.electron:
            for channel in self.beam.channels:
                self.decay_funcs[channel](channel, num_samples, mixing_type)
        elif mixing_type == MixingType.tau:
            self.decay_to_e_pair(num_samples, mixing_type)
        return self

    def decay_to_e_pi(self, channel, num_samples, mixing_type):
        partial_decay = self.__partial_decay_rate_to_electron_pi(mixing_type)
        Logger().log(f"{channel} partial width {partial_decay}")
        self.average_propagation_factor[channel] = self.__get_prop_factor_for_regime(mixing_type, partial_decay)

        electron = Electron()
        pion = Pion()
        electron_rest_momenta = get_two_body_momenta(self, electron, pion, num_samples)
        electron.set_momenta(electron_rest_momenta).boost(self.momenta)
        pion_rest_momenta = get_two_body_momenta(self, pion, electron, num_samples)
        pion.set_momenta(pion_rest_momenta).boost(self.momenta)

        signal = list(zip(electron.momenta, pion.momenta))

        # Apply cuts
        cut_signal = []
        mT_max = 1.85 #GeV
        elec_e_min = 0.8 #GeV
        for elec_p, pion_p in signal:
            p_tot = elec_p + pion_p
            if elec_p.get_energy() > elec_e_min and p_tot.get_transverse_mass() < mT_max:
                cut_signal.append([elec_p, pion_p])
        
        self.efficiency[channel] = len(cut_signal)/len(signal)
        Logger().log(f"{channel} channel efficiency: {self.efficiency[channel]}")

    def decay_to_e_mu(self, channel, num_samples, mixing_type):
        decay_type = DecayType.CCNC

        partial_decay = self.__partial_decay_rate_to_electron_muon(mixing_type, decay_type=decay_type)
        Logger().log(f"{channel} partial width: {partial_decay}")
        self.average_propagation_factor[channel] = self.__get_prop_factor_for_regime(mixing_type, partial_decay)
        
        e_elec = np.linspace(0, self.m/2, 1000)
        e_muon = np.linspace(MUON_MASS, (self.m**2 + MUON_MASS**2)/(2*self.m), 1000, endpoint=False)
        lepton_energy_samples = generate_samples(e_elec, e_muon, \
            dist_func=self.__electron_muon_dist, n_samples=num_samples, \
            region=lambda e_elec, e_muon: e_elec + e_muon > self.m/2)
        
        elec = Electron()
        muon = Muon()
        signal = []
        total_signal_length = min(len(self.momenta), len(lepton_energy_samples))
        # experimental cuts for electron pair
        elec_e_min = 0.8 #GeV
        muon_e_min = 3. #GeV
        mT_max = 1.85 #GeV
        for i in range(total_signal_length):
            momenta = self.__get_lepton_momenta_lab_frame(lepton_energy_samples[i], self.momenta[i], elec, muon)
            if momenta:
                p_elec, p_muon, p_tot = momenta
            else:
                continue

            # apply cuts here
            if p_elec.get_energy() > elec_e_min and p_muon.get_energy() > muon_e_min and p_tot.get_transverse_mass() < mT_max:
                # NOTE we only need this signal if we want to plot
                signal.append([p_elec, p_muon, p_tot]) 

        self.efficiency[channel] = len(signal)/total_signal_length
        Logger().log(f"{channel} channel efficiency: {self.efficiency[channel]}")

    def decay_to_e_pair(self, channel, num_samples, mixing_type):
        # Decay Ne/tau -> e+ e- nu_e/tau
        decay_type = DecayType.CCNC

        partial_decay = self.__partial_decay_rate_to_electron_pair(mixing_type, decay_type=decay_type)
        Logger().log(f"{channel} partial width: {partial_decay}")
        self.average_propagation_factor[channel] = self.__get_prop_factor_for_regime(mixing_type, partial_decay)
        
        e_l_plus = np.linspace(0, self.m/2, 1000)
        e_l_minus = np.linspace(0, self.m/2, 1000)
        lepton_energy_samples = generate_samples(e_l_plus, e_l_minus, dist_func=lambda ep, em: self.__electron_positron_dist_dirac(ep, em, decay_type=decay_type), n_samples=num_samples, \
            region=lambda ep, em: ep + em > self.m/2)
        
        elec1 = Electron()
        elec2 = Electron()
        signal = []
        total_signal_length = min(len(self.momenta), len(lepton_energy_samples))
        # experimental cuts for electron pair
        e_min = 0.8 #GeV
        mT_max = 1.85 #GeV
        for i in range(total_signal_length):
            momenta = self.__get_lepton_momenta_lab_frame(lepton_energy_samples[i], self.momenta[i], elec1, elec2)
            if momenta:
                p1, p2, p_tot = momenta
            else:
                continue

            # apply cuts here
            if p1.get_energy() > e_min and p2.get_energy() > e_min and p_tot.get_transverse_mass() < mT_max:
                # NOTE we only need this signal if we want to plot
                signal.append([p1, p2, p_tot]) 

        self.efficiency[channel] = len(signal)/total_signal_length
        Logger().log(f"{channel} channel efficiency: {self.efficiency[channel]}")
    
    def __get_lepton_momenta_lab_frame(self, lepton_energies, hnl_momentum, lepton1, lepton2):
            e1, e2 = lepton_energies
            e3 = self.m - (e1 + e2)
            p1 = np.sqrt(e1**2 - lepton1.m**2)
            p2 = np.sqrt(e2**2 - lepton2.m**2)
            cos_th_12 = (lepton1.m**2 + lepton2.m**2 + 2*e1*e2 - self.m**2 + 2*self.m*e3)/(2*p1*p2)
            if abs(cos_th_12) > 1:
                Logger().log(f"invalid cos: {cos_th_12}")
                return None
            theta_12 = np.arccos(cos_th_12) # takes values between 0, pi
            theta_1 = np.random.uniform(0, 2*np.pi) # angle of one of the leptons uniformly between 0, 2pi. This is like the cone angle
            theta_2 = theta_1 + theta_12
            cos_theta_1 = np.cos(theta_1)
            cos_theta_2 = np.cos(theta_2)
            p1 = Momentum4.from_polar(e1, cos_theta_1, 0, lepton1.m).boost(hnl_momentum)
            p2 = Momentum4.from_polar(e2, cos_theta_2, 0, lepton2.m).boost(hnl_momentum)
            p_tot = p1 + p2
            return p1,p2,p_tot
    def __apply_cuts(self):
        pass