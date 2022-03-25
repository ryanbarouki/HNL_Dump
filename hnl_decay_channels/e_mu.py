import numpy as np
from decay_type import DecayType
from fundamental_constants import *
from mixing_type import MixingType
from particle_masses import *

class ElectronMuon:
    def __init__(self, beam, parent) -> None:
        self.beam = beam
        self.parent = parent

    def partial_decay_rate(self):
        if self.beam.mixing_type == MixingType.electron:
            # https://arxiv.org/abs/hep-ph/9703333
            xm = MUON_MASS/self.parent.m
            func_form = 1 - 8*xm**2 + 8*xm**6 + xm**8 - 12*xm**4*np.log(xm**2)
            return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3)*func_form)

    def diff_distrubution(self, e_elec, e_muon):
        # https://arxiv.org/abs/hep-ph/9703333
        xm = MUON_MASS/self.parent.m
        x_mu = 2*e_muon/self.parent.m
        return x_mu*(1 - x_mu + xm**2)