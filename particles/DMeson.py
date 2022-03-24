import matplotlib.pyplot as plt
import numpy as np
from .particle import Particle
from .hnl import HNL
from .electron import Electron
from mixing_type import MixingType
from particle_masses import *
from utils import get_two_body_momenta, DEBUG_AVERAGE_MOMENTUM, DEBUG_PLOT_MOMENTA

class DMeson(Particle):
    """D+/- meson"""
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(D_MASS, beam, parent, momenta)

    def decay(self, hnl_mass):
        # D -> N + electron (muon but not implemented)
        if self.beam.mixing_type == MixingType.tau:
            return self

        hnl = HNL(hnl_mass, beam=self.beam, parent=self)
        electron = Electron(parent=self, beam=self.beam)

        # set the kinematics of the children
        hnl_rest_momenta = get_two_body_momenta(self, hnl, electron, num_samples=self.beam.num_samples)

        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta)

        DEBUG_AVERAGE_MOMENTUM(hnl, "Average HNL momentum")
        # DEBUG_PLOT_MOMENTA(hnl, ((0, 200), (0, 3)))

        hnl.decay()

        self.children.append(hnl)
        self.children.append(electron)
        
        return self