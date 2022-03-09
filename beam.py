import numpy as np
from momentum4 import Momentum4
from particles.DsMeson import DsMeson
from particles.DMeson import DMeson
from particle_masses import *
from utils import generate_samples, e_cos_theta_to_momentum4
from mixing_type import MixingType

class BeamExperiment:
    def __init__(self, beam_energy, nucleon_mass, max_opening_angle, detector_length, detector_distance):
        self.s = nucleon_mass*(2*beam_energy + nucleon_mass)
        self.beam_momentum = Momentum4.from_polar(beam_energy + nucleon_mass, 1, 0, np.sqrt(self.s))
        self.beta_cm = beam_energy/(beam_energy + nucleon_mass)
        self.gamma_cm = (beam_energy + nucleon_mass)/np.sqrt(self.s)
        self.children = []
        self.MAX_OPENING_ANGLE = max_opening_angle
        self.DETECTOR_LENGTH = detector_length
        self.DETECTOR_DISTANCE = detector_distance
        self.linear_regime = True
        self.mixing_squared = 1
    
    def with_mixing(self, mixing_squared):
        self.linear_regime = False
        self.mixing_squared = mixing_squared
        return self

    def start_dump(self, hnl_mass, num_samples, mixing_type: MixingType):
        if mixing_type == MixingType.electron:
            self.__D_meson_channel(hnl_mass, num_samples, mixing_type)
            self.__Ds_meson_channel(hnl_mass, num_samples, mixing_type)
        elif mixing_type == MixingType.tau:
            self.__Ds_meson_channel(hnl_mass, num_samples, mixing_type)
        return self

    def __meson_diff_distribution(self, pp, pt2):
        b = 0.93
        n = 6.
        xf = 2*pp/np.sqrt(self.s)
        return np.exp(-b*pt2)*(1 - np.abs(xf))**n

    def __get_meson_kinematics(self, mass, num_samples):
        sqrt_s = np.sqrt(self.s)
        pp = np.linspace(-sqrt_s/2, sqrt_s/2, 1000)
        pt2 = np.linspace(0, self.s/4, 10000)
        samples = generate_samples(pp, pt2, dist_func=self.__meson_diff_distribution, n_samples=num_samples, region=lambda pp, pt2: pp**2 + pt2 < self.s/4 - mass**2)

        momentum4_samples = []
        momentum = 0
        for sample in samples:
            pp, pt2 = sample
            e = np.sqrt(pp**2 + pt2 + mass**2)
            com_momentum = Momentum4.from_e_pt_pp(e, np.sqrt(pt2), pp, 0, mass)
            lab_momentum = com_momentum.boost(-self.beam_momentum)
            momentum += lab_momentum.get_total_momentum()
            momentum4_samples.append(lab_momentum)
        print(f"Average meson momentum: {momentum/len(samples)}")
        return momentum4_samples
    
    def __D_meson_channel(self, hnl_mass, num_samples, mixing_type):
        momentum4_samples = self.__get_meson_kinematics(D_MASS, num_samples)
        D_meson = DMeson(beam=self, momenta=momentum4_samples)
        self.children.append(D_meson)
        D_meson.decay(hnl_mass, num_samples, mixing_type)
    
    def __Ds_meson_channel(self, hnl_mass, num_samples, mixing_type):
        momentum4_samples = self.__get_meson_kinematics(DS_MASS, num_samples)
        Ds_meson = DsMeson(beam=self, momenta=momentum4_samples)
        self.children.append(Ds_meson)
        Ds_meson.decay(hnl_mass, num_samples, mixing_type)  
    
    def find_instances_of_type(self, type, current=None, instances=None):
        if current == None:
            current = self
        if instances == None:
            instances = []
        if isinstance(current, type):
            instances.append(current)
        if current.children:
            for child in current.children:
                instances = self.find_instances_of_type(type, child, instances)
        return instances