from hnl import HNL
from particle import Particle
from particle_masses import *

class Tau(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(TAU_MASS, beam, parent, momenta)
    
    def decay(self):
        # hnl = HNL(1, self.beam, self)
        # self.children.append(hnl)
        pass