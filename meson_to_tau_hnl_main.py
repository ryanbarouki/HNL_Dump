from utils import sliced_hist
from particle import Particle
from beam import BeamExperiment
import numpy as np
import time
import sys

def main():
    if len(sys.argv) < 2:
        raise Exception("Not enough arguments.\nPlease specify the number of samples")
    # -------------------------- set up ---------------------------
    beam = BeamExperiment(beta_cm=0.87, gamma_cm=2, s=52000)
    meson = Ds(m=1.869, beam=beam)
    hnl_from_meson = HNL(m=0.05, beam=beam, parent=meson)
    tau = Particle(m=1.776, beam=beam, parent=meson)
    hnl_from_tau = HNL(m=0.05, beam=beam, parent=tau)

    # -------------------------- sampling ---------------------------
    num_samples = int(sys.argv[1])
    start = time.time()
    hnl_meson_rest_sample = hnl_from_meson.get_two_body_dist_samples(other_particle=tau, sample_range=(0., 1.), n=num_samples)
    tau_rest_sample = tau.get_two_body_dist_samples(other_particle=hnl_from_meson, sample_range=(0., 1.), n=num_samples)
    e_max = (tau.m**2 + hnl_from_tau.m**2)/(2*tau.m)
    hnl_tau_rest_sample = hnl_from_tau.get_three_body_tau_dist_samples(sample_range=((hnl_from_tau.m, e_max), (0., 1.)), n=num_samples)
    meson_sample = meson.get_dist_samples(sample_range=((meson.m, np.sqrt(beam.s)), (0., 1.)), n=num_samples)

    # -------------------------- boost to lab frame ---------------------------
    hnl_from_meson_lab = hnl_from_meson.lab_dist(hnl_meson_rest_sample, meson_sample)
    tau_lab_dist = tau.lab_dist(tau_rest_sample, meson_sample)
    hnl_from_tau_lab = hnl_from_tau.lab_dist(hnl_tau_rest_sample, tau_lab_dist)
    end = time.time()
    print("Time taken: {}".format(end - start))

    # --------------------------- write to file -------------------------
    f = open("hnl_from_tau_lab_dist_n={}.out".format(num_samples), "w")
    for arr in hnl_from_tau_lab:
        f.write("{},{}\n".format(arr[0], arr[1]))
    f.close()

if __name__ == "__main__":
    main()