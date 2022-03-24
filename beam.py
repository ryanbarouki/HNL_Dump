import numpy as np
from experimental_constants import *
from momentum4 import Momentum4
from particles.DsMeson import DsMeson
from particles.DMeson import DMeson
from particle_masses import *
from utils import generate_samples, e_cos_theta_to_momentum4
from mixing_type import MixingType
from logger import Logger
class BeamExperiment:
    def __init__(self, mixing_type, num_samples):
        self.s = NUCLEON_MASS*(2*BEAM_ENERGY + NUCLEON_MASS)
        self.beam_momentum = Momentum4.from_polar(BEAM_ENERGY + NUCLEON_MASS, 1, 0, np.sqrt(self.s))
        self.beta_cm = BEAM_ENERGY/(BEAM_ENERGY + NUCLEON_MASS)
        self.gamma_cm = (BEAM_ENERGY + NUCLEON_MASS)/np.sqrt(self.s)
        self.children = []
        self.max_opening_angle = DETECTOR_OPENING_ANGLE
        self.detector_length = DETECTOR_LENGTH
        self.detector_distance = DETECTOR_DISTANCE
        self.linear_regime = True
        self.mixing_squared = 1
        self.mixing_type = mixing_type
        self.num_samples = num_samples
        if mixing_type == MixingType.electron:
            self.channels = ELECTRON_HNL_CHANNELS
        elif mixing_type == MixingType.tau:
            self.channels = TAU_HNL_CHANNELS
    
    def with_mixing(self, mixing_squared):
        self.linear_regime = False
        self.mixing_squared = mixing_squared
        return self

    def start_dump(self, hnl_mass):
        if self.mixing_type == MixingType.electron:
            self.__D_meson_channel(hnl_mass)
            self.__Ds_meson_channel(hnl_mass)
        elif self.mixing_type == MixingType.tau:
            self.__Ds_meson_channel(hnl_mass)
        return self

    def __meson_diff_distribution(self, pp, pt2):
        b = 0.93
        n = 6.
        xf = 2*pp/np.sqrt(self.s)
        return np.exp(-b*pt2)*(1 - np.abs(xf))**n

    def __get_meson_kinematics(self, mass):
        sqrt_s = np.sqrt(self.s)
        pp = np.linspace(-sqrt_s/2, sqrt_s/2, 1000)
        pt2 = np.linspace(0, self.s/4, 10000)
        samples = generate_samples(pp, pt2, dist_func=self.__meson_diff_distribution, n_samples=self.num_samples, region=lambda pp, pt2: pp**2 + pt2 < self.s/4 - mass**2)

        momentum4_samples = []
        momentum = 0
        for sample in samples:
            pp, pt2 = sample
            e = np.sqrt(pp**2 + pt2 + mass**2)
            com_momentum = Momentum4.from_e_pt_pp(e, np.sqrt(pt2), pp, 0, mass)
            lab_momentum = com_momentum.boost(-self.beam_momentum)
            momentum += lab_momentum.get_total_momentum()
            momentum4_samples.append(lab_momentum)
        Logger().log(f"Average meson momentum: {momentum/len(samples)}")
        return momentum4_samples
    
    def __D_meson_channel(self, hnl_mass):
        momentum4_samples = self.__get_meson_kinematics(D_MASS)
        D_meson = DMeson(beam=self, momenta=momentum4_samples)
        self.children.append(D_meson)
        D_meson.decay(hnl_mass)
    
    def __Ds_meson_channel(self, hnl_mass):
        momentum4_samples = self.__get_meson_kinematics(DS_MASS)
        Ds_meson = DsMeson(beam=self, momenta=momentum4_samples)
        self.children.append(Ds_meson)
        Ds_meson.decay(hnl_mass)  
    
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