import numpy as np
from .hnl import HNL
from .particle import Particle
from .pion import Pion
from particle_masses import *
from utils import generate_samples, e_cos_theta_to_momentum4, get_two_body_momenta
from .tau_decay_modes import TauDecayModes

class Tau(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(TAU_MASS, beam, parent, momenta)
    
    def decay(self, hnl_mass):
        #TODO make sure the masses make sense for the decays
        hnl = HNL(hnl_mass, beam=self.beam, parent=self, decay_mode=TauDecayModes.hnl_lepton_nu)
        hnl2body = HNL(hnl_mass, beam=self.beam, parent=self, decay_mode=TauDecayModes.hnl_pi)
        pion = Pion(beam=self.beam, parent=self)
        e_max = (self.m**2 + hnl_mass**2)/(2*self.m)
        e = np.linspace(hnl_mass, e_max, 1_000)
        cos_theta = np.linspace(0., 1., 1_000)
        hnl_rest_samples = generate_samples(e, cos_theta, dist_func=lambda e, cos: self.diff_decay_to_hnl_nu_lepton(hnl_mass, e, cos), n_samples=self.beam.num_samples)
        hnl_rest_momenta  = e_cos_theta_to_momentum4(hnl_rest_samples, hnl_mass)

        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta)

        hnl_rest_momenta_two_body = get_two_body_momenta(self, hnl, pion, self.beam.num_samples)
        hnl2body.set_momenta(hnl_rest_momenta_two_body).boost(self.momenta)

        hnl.decay()
        hnl2body.decay()
        self.children.append(hnl)
        self.children.append(hnl2body)
        return self

    def diff_decay_to_hnl_nu_lepton(self, hnl_mass, e, cos_theta):
        # valid for all leptons because the lepton mass has been neglected
        aux0=3.+(((-4.*e)/self.m)+(((0.5*((hnl_mass**2)*(-4.+((6.*e)/self.m))))/self.m)/e))
        output=8.*((e**2)*((np.sqrt((1.-((e**-2.)*(hnl_mass**2)))))*(aux0*(self.m**-3.))))
        return output