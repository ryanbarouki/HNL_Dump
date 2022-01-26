
import numpy as np
from particle_masses import DS_MASS
from utils import generate_samples, e_cos_theta_to_momentum4
from particle import Particle
from electron import Electron
from muon import Muon
from tau import Tau
from hnl import HNL
from mixing_type import MixingType

class DsMeson(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(DS_MASS, beam, parent, momenta)

    def decay(self, hnl_mass, num_samples, mixing_type: MixingType):
        # D -> N + lepton (electron, muon, tau)
        hnl = HNL(hnl_mass, beam=self.beam, parent=self)
        lepton = None
        if mixing_type == MixingType.electron:
            lepton = Electron(parent=self, beam=self.beam)
        elif mixing_type == MixingType.muon:
            lepton = Muon(parent=self, beam=self.beam)
        elif mixing_type == MixingType.tau:
            lepton = Tau(parent=self, beam=self.beam)

        # set the kinematics of the children
        e0_hnl = (self.m**2 + hnl.m**2 - lepton.m**2) / (2*self.m)
        e_hnl = np.full(1000, e0_hnl)
        e0_lepton = (self.m**2 + lepton.m**2 - hnl.m**2) / (2*self.m)
        e_lepton = np.full(1000, e0_lepton)
        cos = np.linspace(0., 1., 1000)
        unit_func = lambda e, cos: e/e
        hnl_rest_samples = generate_samples(e_hnl, cos, dist_func=unit_func, n_samples=num_samples)
        hnl_rest_momenta = e_cos_theta_to_momentum4(hnl_rest_samples, hnl.m)
        lepton_rest_samples = generate_samples(e_lepton, cos, dist_func=unit_func, n_samples=num_samples)
        lepton_rest_momenta = e_cos_theta_to_momentum4(lepton_rest_samples, lepton.m)

        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta) \
                                         .geometric_cut(0, self.beam.MAX_OPENING_ANGLE)
        lepton.set_momenta(lepton_rest_momenta).boost(self.momenta)

        self.children.append(hnl)
        self.children.append(lepton)

        hnl.decay(num_samples, mixing_type)
        lepton.decay()
        
        return self