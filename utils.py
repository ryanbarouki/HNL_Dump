import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from momentum4 import Momentum4
from logger import Logger

def DEBUG_PLOT_MOMENTA(particle, range):
    samples = []
    for momentum in particle.momenta:
        pt = momentum.get_transverse_momentum()
        pp = momentum.get_parallel_momentum()
        samples.append([pp, pt])
    samples = np.array(samples)

    fig = plt.figure()
    plt.hist2d(samples[:,0], samples[:,1], bins=100, range=range)
    plt.show()

def DEBUG_AVERAGE_MOMENTUM(particle, text):
    total_p = 0
    for momentum in particle.momenta:
        total_p += momentum.get_total_momentum()
    Logger().log(f"{text}: {total_p/len(particle.momenta)}")

def e_cos_theta_to_momentum4(samples, mass):
    momentum4_samples = []
    for sample in samples:
        e, cos_theta = sample
        momentum4_samples.append(Momentum4.from_polar(e, cos_theta, 0, mass))
    return np.array(momentum4_samples)

def e_cos_theta_phi_to_momentum4(samples, mass):
    momentum4_samples = []
    for sample in samples:
        e, cos_theta, phi = sample
        momentum4_samples.append(Momentum4.from_polar(e, cos_theta, phi, mass))
    return np.array(momentum4_samples)
    
def get_two_body_momenta(parent, particle, other_particle, num_samples, in_plane=False):
    e0 = (parent.m**2 + particle.m**2 - other_particle.m**2) / (2*parent.m)
    samples = []
    for i in range(num_samples):
        cos = np.random.uniform(0.0, 1.0)
        phi = 0 if in_plane else np.random.uniform(0, 2*np.pi)
        samples.append([e0, cos, phi])
    sample_momenta = e_cos_theta_phi_to_momentum4(samples, particle.m)
    return sample_momenta

def sample_2d_dist(dist_func, x, y, n):
    X, Y = np.meshgrid(x, y)
    dist = dist_func(X, Y) 
    dist = dist / dist.sum()
    # Create a flat copy of the array
    flat = dist.flatten()
    # Then, sample an index from the 1D array with the
    # probability distribution from the original array
    sample_index = np.random.choice(a=flat.size, p=flat, size=n)
    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, dist.shape)
    x_samples = x[adjusted_index[1]]
    y_samples = y[adjusted_index[0]]
    return np.array(list(zip(x_samples, y_samples)))

def generate_samples(*x, dist_func, n_samples, region=lambda *x: True) -> np.ndarray:
    X = np.meshgrid(*x)
    dist = dist_func(*X)
    # If the points are outside the region then dist = 0 at that point
    dist = np.where(region(*X), dist, np.zeros(dist.shape))
    dist = dist / dist.sum()
    # Create a flat copy of the array
    flat = dist.flatten()
    # Then, sample an index from the 1D array with the
    # probability distribution from the original array
    sample_index = np.random.choice(a=flat.size, p=flat, size=n_samples)
    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, dist.shape)
    samples = []
    # Pick out the coordinate values from the indices
    for i in range(len(x)):
        samples.append(x[i][adjusted_index[len(x)-1-i]])
    # Zip them up in (x, y, ...) tuples
    return np.array(list(zip(*samples)))

def plot_surface(func, x, y, range):
    x_range, y_range = range
    x_min, x_max = x_range
    y_min, y_max = y_range
    x = x[np.where(x <= x_max)]
    x = x[np.where(x >= x_min)]
    y = y[np.where(y >= y_min)]
    y = y[np.where(y <= y_max)]
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, func(X,Y))
    plt.show()

def sliced_hist(samples_2d, var_to_fix, cut):
    lower, upper = cut
    sliced = samples_2d[np.where(samples_2d[:,var_to_fix] < upper)]
    sliced = sliced[np.where(sliced[:,var_to_fix] > lower)]
    var = 1 - var_to_fix
    return sliced[:, var]

def write_to_file_csv(samples, filepath):
    f = open(filepath, "w")
    for arr in samples:
        row = ""
        for i in range(len(arr)):
            row += f"{arr[i]}"
            row += "," if i != len(arr) - 1 else "\n"
        f.write(row)
    f.close()