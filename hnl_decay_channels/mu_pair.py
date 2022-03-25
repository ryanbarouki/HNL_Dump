import numpy as np
from fundamental_constants import *

class MuonPair:
    def __init__(self, beam, parent) -> None:
        self.beam = beam
        self.parent = parent

    def partial_decay_rate(self):
        # Boiarska et al
        c1 = 0.25*(1 - 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        return self.beam.mixing_squared*(GF**2*self.m**5/(192*np.pi**3))*c1

    def diff_distribution(self, e_plus, e_minus):
        #TODO find and implement
        # https://arxiv.org/abs/hep-ph/9703333 not sure if the one there is correct for electrons and tau
        pass
