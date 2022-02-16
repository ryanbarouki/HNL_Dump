import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from momentum4 import Momentum4

def e_cos_theta_to_momentum4(samples, mass):
    momentum4_samples = []
    for sample in samples:
        e, cos_theta = sample
        momentum4_samples.append(Momentum4.from_polar(e, cos_theta, 0, mass))
    return np.array(momentum4_samples)
    
def get_two_body_momenta(parent, particle, other_particle, num_samples):
    e0 = (parent.m**2 + particle.m**2 - other_particle.m**2) / (2*parent.m)
    e = np.full(1000, e0)
    cos = np.linspace(0., 1., 1000)
    unit_func = lambda e, cos: e/e
    samples = generate_samples(e, cos, dist_func=unit_func, n_samples=num_samples)
    sample_momenta = e_cos_theta_to_momentum4(samples, particle.m)
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