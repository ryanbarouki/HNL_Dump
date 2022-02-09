import numpy as np
from particle import Particle
from hnl import HNL
from electron import Electron
from mixing_type import MixingType
from particle_masses import *
from utils import generate_samples, e_cos_theta_to_momentum4

class DMeson(Particle):
    """D+/- meson"""
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(D_MASS, beam, parent, momenta)

    def decay(self, hnl_mass, num_samples, mixing_type: MixingType):
        # D -> N + electron (muon but not implemented)
        hnl = HNL(hnl_mass, beam=self.beam, parent=self)
        electron = Electron(parent=self, beam=self.beam)

        # set the kinematics of the children
        hnl_rest_momenta = self.__get_two_body_momenta(hnl, electron, num_samples)

        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta) \
                                         .geometric_cut(0, self.beam.MAX_OPENING_ANGLE)

        hnl.decay(num_samples, mixing_type)

        self.children.append(hnl)
        self.children.append(electron)
        
        return self
    
    def __get_two_body_momenta(self, particle, other_particle, num_samples):
        e0 = (self.m**2 + particle.m**2 - other_particle.m**2) / (2*self.m)
        e = np.full(1000, e0)
        cos = np.linspace(0., 1., 1000)
        unit_func = lambda e, cos: e/e
        samples = generate_samples(e, cos, dist_func=unit_func, n_samples=num_samples)
        sample_momenta = e_cos_theta_to_momentum4(samples, particle.m)
        return sample_momenta