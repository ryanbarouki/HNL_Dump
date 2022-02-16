import numpy as np
from particle_masses import D_MASS
from beam import BeamExperiment
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
from utils import sample_2d_dist, plot_surface, generate_samples
from constants import *

def gauss(x, y):
    return np.exp(-10*(x*x + y*y))

def gauss_1(x):
    a = 100
    dist = np.exp(-a*x*x)
    return 2*dist / np.sqrt(np.pi/a)

def AR(func, n):
    count = 0
    samples = []
    while count < n:
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)

        f = func(x, y)
        u = np.random.uniform()
        if f > u:
            samples.append([x,y])
            count += 1
    return np.array(samples)

def sample_non_uniform(func, n):
    x = np.linspace(0, 1, 1000)
    non_uniform_x = x**4
    f = lambda x: 4*x**3*func(x**4)
    dist = f(x)
    p = dist / dist.sum()

    # without the derivative jacobian
    dist2 = func(non_uniform_x)
    p2 = dist2 / dist2.sum()
    samples = np.random.choice(a=non_uniform_x, p=p, size=n)
    return samples

def sample_uniform(func, n):
    x = np.linspace(0, 1, 1000)
    dist = func(x)
    p = dist / dist.sum()
    samples = np.random.choice(a=x, p=p, size=n)
    return samples

# beam = beam(beta_cm=0.87, gamma_cm=2, s=52000)
# meson = Meson(m=1.8, beam=beam)
# samples_non_uniform = sample_non_uniform(gauss_1, 10000)
# samples_uniform = sample_uniform(gauss_1, 10000)

# plt.figure(0)
# plt.hist(samples_non_uniform, bins=50, density=True)
# plt.hist(samples_uniform, bins=50, density=True, alpha=0.7)
# plt.plot(np.linspace(0, 1, 1000), gauss_1(np.linspace(0, 1, 1000)))

# e = np.linspace(meson.m, np.sqrt(beam.s), 1000)
# cos =  np.linspace(0, 1, 1000)
# plot_surface(meson.dist, e, cos, range=((meson.m, np.sqrt(beam.s)), (0., 1)))

beam = BeamExperiment(beam_energy=400, nucleon_mass=1.0, \
    max_opening_angle=DETECTOR_OPENING_ANGLE, detector_length=DETECTOR_LENGTH, \
    detector_distance=DETECTOR_DISTANCE)

# boosted
momenta = beam.__get_meson_kinematics(D_MASS, 10000)
sqrt_s = np.sqrt(beam.s)
samples = []
for momentum in momenta:
    pt = momentum.get_transverse_momentum()
    pp = momentum.get_parallel_momentum()
    samples.append([pp, pt])
samples = np.array(samples)

fig = plt.figure()
plt.hist2d(samples[:,0], samples[:,1], bins=100, range=((0, 200), (0,6)))

# non-boosted
pp = np.linspace(-sqrt_s/2, sqrt_s/2, 1000)
pt2 = np.linspace(0, beam.s/4, 10000)
samples = generate_samples(pp, pt2, dist_func=beam.__meson_diff_distribution, n_samples=10000)

fig2 = plt.figure()
plt.hist2d(samples[:,0], samples[:,1], bins=100, range=((-sqrt_s/2, sqrt_s/2), (0,10)))

plt.show()
