from .neutrino import Neutrino
from particle_masses import *
from utils import get_two_body_momenta
from .particle import Particle
from .electron import Electron
from .tau import Tau
from .hnl import HNL
from mixing_type import MixingType

class DsMeson(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(DS_MASS, beam, parent, momenta)

    def __decay_tau_mixing(self, hnl_mass, num_samples, mixing_type):
        tau = Tau(parent=self, beam=self.beam)
        other_particle = Neutrino(parent=self, beam=self.beam)
        if TAU_MASS + hnl_mass < DS_MASS:
            # HNL mass is small enough to produce an HNL here
            other_particle = HNL(hnl_mass, beam=self.beam, parent=self)
        
        tau_rest_momenta = self.__get_two_body_momenta(tau, other_particle, num_samples)

        tau.set_momenta(tau_rest_momenta).boost(self.momenta)

        tau.decay(hnl_mass, num_samples, mixing_type)

        if isinstance(other_particle, HNL):
            hnl_rest_momenta = self.__get_two_body_momenta(other_particle, tau, num_samples)
            other_particle.set_momenta(hnl_rest_momenta).boost(self.momenta) \
                                         .geometric_cut(0, self.beam.MAX_OPENING_ANGLE)
            other_particle.decay(num_samples, mixing_type)

        self.children.append(other_particle)
        self.children.append(tau)
        

    def __decay_electron_mixing(self, hnl_mass, num_samples, mixing_type):
        electron = Electron(parent=self, beam=self.beam)
        hnl = HNL(hnl_mass, beam=self.beam, parent=self)
        
        hnl_rest_momenta = get_two_body_momenta(self, hnl, electron, num_samples)
        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta) \
                                        .geometric_cut(0, self.beam.MAX_OPENING_ANGLE)
        hnl.decay(num_samples, mixing_type)

        self.children.append(hnl)
        self.children.append(electron)

    def decay(self, hnl_mass, num_samples, mixing_type: MixingType):
        # D -> N + lepton (electron, muon, tau)
        if mixing_type == MixingType.electron:
            self.__decay_electron_mixing(hnl_mass, num_samples, mixing_type)
        elif mixing_type == MixingType.tau:
            self.__decay_tau_mixing(hnl_mass, num_samples, mixing_type)
        return self