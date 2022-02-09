import numpy as np
from hnl import HNL
from particle import Particle
from particle_masses import *
from utils import generate_samples, e_cos_theta_to_momentum4

class Tau(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(TAU_MASS, beam, parent, momenta)
    
    def decay(self, hnl_mass, num_samples, mixing_type):
        hnl = HNL(hnl_mass, beam=self.beam, parent=self)
        e_max = (self.m**2 + hnl_mass**2)/(2*self.m)
        e = np.linspace(hnl_mass, e_max, 1000)
        cos_theta = np.linspace(0., 1., 1000)
        hnl_rest_samples = generate_samples(e, cos_theta, dist_func=lambda e, cos: self.diff_decay_to_hnl_nu_lepton(hnl_mass, e, cos), n_samples=num_samples)
        hnl_rest_momenta  = e_cos_theta_to_momentum4(hnl_rest_samples, hnl_mass)

        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta)

        hnl.decay(num_samples=num_samples, mixing_type=mixing_type)
        self.children.append(hnl)
        return self

    def diff_decay_to_hnl_nu_lepton(self, hnl_mass, e, cos_theta):
        # valid for all leptons because the lepton mass has been neglected
        aux0=3.+(((-4.*e)/self.m)+(((0.5*((hnl_mass**2)*(-4.+((6.*e)/self.m))))/self.m)/e))
        output=8.*((e**2)*((np.sqrt((1.-((e**-2.)*(hnl_mass**2)))))*(aux0*(self.m**-3.))))
        return output