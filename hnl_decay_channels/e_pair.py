import numpy as np
from decay_type import DecayType
from fundamental_constants import *
from mixing_type import MixingType

class ElectronPair:
    def __init__(self, beam, parent) -> None:
        self.beam = beam
        self.parent = parent

    def partial_decay_rate(self, decay_type=DecayType.CCNC):
        # https://arxiv.org/pdf/2109.03831.pdf
        c1 = 0.25*(1 - 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        c2 = 0.25*(1 + 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        if self.beam.mixing_type == MixingType.electron:
            # Ne -> e+ e- nu_e
            if decay_type == DecayType.CC:
                return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3))
            elif decay_type == DecayType.CCNC:
                return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3))*c2
            else:
                raise Exception("No value for only neutral current")
        elif self.beam.mixing_type == MixingType.tau:
            # N_tau -> e+ e- nu_tau
            return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3))*c1

    def diff_distribution(self, ep, em, decay_type=DecayType.CCNC):
        gr = SIN_WEINB**2 - 1/2
        gl = SIN_WEINB**2
        if decay_type == DecayType.CC:
            gr = 0
            gl = 0
        elif decay_type == DecayType.NC:
            gl = gl - 1
        output = gr**2*em*(self.parent.m - 2*em) + (1-gl)**2*ep*(self.parent.m - 2*ep)
        return output