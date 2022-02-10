import numpy as np
from DsMeson import DsMeson
from DMeson import DMeson
from particle_masses import *
from utils import generate_samples, e_cos_theta_to_momentum4
from mixing_type import MixingType

class BeamExperiment:
    def __init__(self, beam_energy, nucleon_mass, max_opening_angle, detector_length, detector_distance):
        self.s = nucleon_mass*(2*beam_energy + nucleon_mass)
        self.beta_cm = beam_energy/(beam_energy + nucleon_mass)
        self.gamma_cm = (beam_energy + nucleon_mass)/np.sqrt(self.s)
        self.children = []
        self.MAX_OPENING_ANGLE = max_opening_angle
        self.DETECTOR_LENGTH = detector_length
        self.DETECTOR_DISTANCE = detector_distance
        self.linear_regime = True
        self.mixing_squared = 1
    
    def with_mixing(self, mixing_squared):
        self.linear_regime = False
        self.mixing_squared = mixing_squared

    def start_dump(self, hnl_mass, num_samples, mixing_type: MixingType):
        if mixing_type == MixingType.electron:
            # I think Ds decays contribute here too? But I think Subir ignores this
            self.__D_meson_channel(hnl_mass, num_samples, mixing_type)
        elif mixing_type == MixingType.tau:
            self.__Ds_meson_channel(hnl_mass, num_samples, mixing_type)
        return self

    def __meson_diff_distribution(self, mass, eM, c_thetaM):
        # params
        b = 0.93
        n = 6.
        gamma_cm = self.gamma_cm
        beta_cm = self.beta_cm
        s = self.s

        aux0=np.exp(((-b*((((1.-(c_thetaM**2))**2))*((((eM**2)-(mass**2))**2))))))
        aux1=(eM*(np.sqrt(((eM**2)-(mass**2)))))+(c_thetaM*((mass**2)*beta_cm))
        aux2=np.abs(((s**-0.5)*((aux1-(c_thetaM*((eM**2)*beta_cm)))*gamma_cm)))
        aux3=(c_thetaM*((np.sqrt(((eM**2)-(mass**2))))*gamma_cm))-(eM*(beta_cm*gamma_cm))
        output=4.*(aux0*(aux2*((1.+(-2.*(np.abs(((s**-0.5)*aux3)))))**n)))
        return output
    
    def __D_meson_channel(self, hnl_mass, num_samples, mixing_type):
        momentum4_samples = self.__get_meson_kinematics(D_MASS, num_samples)
        D_meson = DMeson(beam=self, momenta=momentum4_samples)
        self.children.append(D_meson)
        D_meson.decay(hnl_mass, num_samples, mixing_type)
    
    def __Ds_meson_channel(self, hnl_mass, num_samples, mixing_type):
        momentum4_samples = self.__get_meson_kinematics(DS_MASS, num_samples)
        Ds_meson = DsMeson(beam=self, momenta=momentum4_samples)
        self.children.append(Ds_meson)
        Ds_meson.decay(hnl_mass, num_samples, mixing_type)  
    
    def __get_meson_kinematics(self, mass, num_samples):
        e_m = np.linspace(mass, np.sqrt(self.s), 10000)
        cos_th_m = np.linspace(0., 1., 10000)
        # Specific non-linear sampling for meson distribution to sample points closer to 1 more finely
        x = -1.0*(cos_th_m - 1)**4 + 1
        # Note the need for the Jacobian factor
        stretched_dist = lambda e, x: (-4*(1-x)**0.75)*self.__meson_diff_distribution(mass, e, x)
        samples = generate_samples(e_m, x, dist_func=stretched_dist, n_samples=num_samples)
        momentum4_samples = e_cos_theta_to_momentum4(samples, mass)
        return momentum4_samples
    
    def find_instances_of_type(self, type, current=None, instances=[]):
        if current == None:
            current = self
        if isinstance(current, type):
            instances.append(current)
        if current.children:
            for child in current.children:
                instances = self.find_instances_of_type(type, child, instances)
        return instances