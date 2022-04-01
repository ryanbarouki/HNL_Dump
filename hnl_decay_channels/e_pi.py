import numpy as np
from fundamental_constants import *
from mixing_type import MixingType
from particle_masses import *
from particles.electron import Electron
from particles.pion import Pion
from utils import get_two_body_momenta

class ElectronPion:
    def __init__(self, beam, parent) -> None:
        self.beam = beam
        self.parent = parent

    def partial_decay_rate(self):
        if self.beam.mixing_type == MixingType.electron:
            phase_factor = 1 # TODO find this factor https://arxiv.org/pdf/1805.08567.pdf
            return self.beam.mixing_squared*(GF**2*self.parent.m**3*F_PI**2/(16*np.pi))*phase_factor

    def decay(self):
        electron = Electron()
        pion = Pion()
        electron_rest_momenta = get_two_body_momenta(self.parent, electron, pion, self.beam.num_samples)
        electron.set_momenta(electron_rest_momenta).boost(self.parent.momenta)
        pion_rest_momenta = get_two_body_momenta(self.parent, pion, electron, self.beam.num_samples)
        pion.set_momenta(pion_rest_momenta).boost(self.parent.momenta)

        signal = list(zip(electron.momenta, pion.momenta))

        # Apply cuts
        cut_signal = []
        mT_max = 1.85 #GeV
        elec_e_min = 0.8 #GeV
        for elec_p, pion_p in signal:
            p_tot = elec_p + pion_p
            if elec_p.get_energy() > elec_e_min and p_tot.get_transverse_mass() < mT_max:
                cut_signal.append([elec_p, pion_p])
        
        efficiency= len(cut_signal)/len(signal)
        return efficiency
