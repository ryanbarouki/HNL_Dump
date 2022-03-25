import numpy as np
from fundamental_constants import *
from mixing_type import MixingType
from particle_masses import *

class ElectronPion:
    def __init__(self, beam, parent) -> None:
        self.beam = beam
        self.parent = parent

    def partial_decay_rate(self):
        if self.beam.mixing_type == MixingType.electron:
            phase_factor = 1 # TODO find this factor https://arxiv.org/pdf/1805.08567.pdf
            return self.beam.mixing_squared*(GF**2*self.parent.m**3*F_PI**2/(16*np.pi))*phase_factor