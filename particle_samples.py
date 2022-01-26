from momentum4 import Momentum4
from particle import Particle
import numpy as np
from utils import sample_nd_dist

class ParticleSamples:
    """Many methods in this class assume particle samples are in (E, cos(angle to beam)) format"""
    def __init__(self, particle, decay_dist) -> None:
        self.particle = particle
        self.decay_dist = decay_dist
        self.samples = np.array([])

    def generate_samples(self, *x, n_samples):
        self.samples = sample_nd_dist(*x, dist_func=self.decay_dist, n_samples=n_samples)
        return self
    
    def add_propagation_factor(self, L, decay_rate):
        """Assumes that the samples are in the form (E, cos(theta))"""
        if self.samples.size == 0:
            raise Exception("Samples have not been generated")
        new_samples = []
        for sample in self.samples:
            e, cos = sample
            p = np.sqrt(e**2 - self.particle.m**2)
            factor = L*self.particle.m*decay_rate/p
            # if factor > 1:
            #     raise Exception("Factor is greater than 1!")
            new_samples.append([*sample, factor])

        propagated_samples = ParticleSamples(self.particle, self.decay_dist)
        propagated_samples.samples = np.array(new_samples)
        return propagated_samples

    def propagate_linear(self, L, decay_rate):
        """Assumes that the samples are in the form (E, cos(theta))"""
        if self.samples.size == 0:
            raise Exception("Samples have not been generated")
        new_samples = []
        for sample in self.samples:
            e, cos = sample
            p = np.sqrt(e**2 - self.particle.m**2)
            factor = L*self.particle.m*decay_rate/p
            if factor > 1:
                raise Exception("Factor is greater than 1!")
            uniform = np.random.uniform()
            if uniform < factor:
                new_samples.append(sample)

        if len(new_samples) == 0:
            raise Exception("No samples survived propagation step")

        propagated_samples = ParticleSamples(self.particle, self.decay_dist)
        propagated_samples.samples = np.array(new_samples)
        return propagated_samples
        
    def boost(self, parent_sample):
        length = min(len(self.samples), len(parent_sample))
        lab_samples = []
        for i in range(length):
            p_parent = Momentum4.from_polar(parent_sample[i,0], parent_sample[i,1], 0, self.particle.parent.m) 
            p_child_rest = Momentum4.from_polar(self.samples[i,0], self.samples[i,1], 0, self.particle.m)
            p_child_lab = p_child_rest.boost(-p_parent)
            lab_samples.append([p_child_lab.get_energy(), p_child_lab.get_cos_theta()])
        lab_particle_samples = ParticleSamples(self.particle, self.decay_dist)
        lab_particle_samples.samples = np.array(lab_samples)
        return lab_particle_samples
    
    def append(self, sample):
        if self.samples.size == 0:
            self.samples = np.append([self.samples], [sample], axis=1)
        else:
            self.samples = np.append(self.samples, [sample], axis=0)

    def geometric_cut(self, min_angle, max_angle):
        max_cos_theta = np.cos(min_angle)
        min_cos_theta = np.cos(max_angle)
        cut_samples = self.samples[np.where(self.samples[:,1] < max_cos_theta)]
        cut_samples = cut_samples[np.where(cut_samples[:,1] > min_cos_theta)]
        cut_particle_samples = ParticleSamples(self.particle, self.decay_dist)
        cut_particle_samples.samples = np.array(cut_samples)
        return cut_particle_samples