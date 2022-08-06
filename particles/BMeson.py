from .particle import Particle
from particle_masses import B_MASS

class BMeson(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(B_MASS, beam, parent, momenta)
    
    def decay(self, hnl_mass):
        pass