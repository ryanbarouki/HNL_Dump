import matplotlib.pyplot as plt
from particle_samples import ParticleSamples
from utils import sliced_hist, write_to_file_csv
from particles import Particle, HNL, Ds, Lepton, DecayType
import cross_sections as CS
import branching_ratios as BR
from beam import BeamExperiment
import numpy as np
import time
import sys

DETECTOR_WIDTH = 1.85 # metres

def get_di_lepton_lab_frame(lepton_samples: ParticleSamples, hnl_lab_samples: ParticleSamples) -> ParticleSamples:
    """Returns the total energy and total transverse momentum of lepton pair"""
    nu = Particle(0, lepton_samples.particle.beam, lepton_samples.particle.parent)
    # Sum of lepton pair momentum is anti-parallel to neutrino momentum
    nu_samples = ParticleSamples(particle=nu, decay_dist=lepton_samples.decay_dist)
    for sample in lepton_samples.samples:
        e_plus, e_minus = sample
        hnl_mass = lepton_samples.particle.parent.m
        # Valid sample region is a triangle
        if e_plus + e_minus <= hnl_mass/2:
            continue
        e_nu = hnl_mass - (e_plus + e_minus) # Energy of the neutrino in rest frame
        cos_th = np.random.uniform()
        nu_samples.append([e_nu, cos_th])
    # Cut the samples short since the sample region constraint will remove some points
    hnl_lab_samples_cut = hnl_lab_samples.samples[:len(nu_samples.samples)]
    nu_lab_samples = nu_samples.boost(hnl_lab_samples_cut)
    nu_lab_energies =  nu_lab_samples.samples[:,0]
    nu_lab_cos_theta = nu_lab_samples.samples[:,1]
    hnl_decay_factors = hnl_lab_samples_cut[:,2]
    # p+ + p- has same momentum distribution as the neutrino in rest frame but energy is not the same
    # So we must get the energy of the lepton pair in the lab frame
    di_lepton_energies = hnl_lab_samples_cut[:,0] - nu_lab_energies
    di_lepton_trans_momentum = nu_lab_energies*np.sqrt(1-nu_lab_cos_theta**2)
    return np.array(list(zip(di_lepton_energies, di_lepton_trans_momentum, hnl_decay_factors)))

def apply_BEBC_cuts(di_lepton_samples: np.ndarray, e_min, pT_max) -> np.ndarray:
    """Assumes that samples are in the form (total Energy, total pT) of lepton pair"""
    cut_lepton_samples = di_lepton_samples[np.where(di_lepton_samples[:,0] > e_min)]
    cut_lepton_samples = cut_lepton_samples[np.where(cut_lepton_samples[:,1] < pT_max)]
    return cut_lepton_samples

def get_total_normalised_hnl_decays(cut_samples: np.ndarray, num_samples, hnl_mass):
    if len(cut_samples[0]) < 3:
        raise Exception("Samples don't have required format")
    factors = cut_samples[:,2]
    fraction_decay = sum(factors) / num_samples
    print(f"fraction decay: {fraction_decay}")
    flux_density = 32e-9 # electron neutrinos per proton per micro sr
    opening_angle = 46 # micro sr
    pot = 1.9e18 # protons on target
    electron_nu_massless_flux = flux_density*pot*opening_angle
    print(f"Muon nu massless flux: {electron_nu_massless_flux}")
    normalisation = electron_nu_massless_flux*(CS.P_TO_DPDM_X*BR.D_TO_E_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)
    normalisation_to_muon = electron_nu_massless_flux*(CS.P_TO_DPDM_X*BR.D_TO_MU_HNL(hnl_mass))/(CS.P_TO_DPDM_X*BR.D_TO_MU_NUMU_X + CS.P_TO_D0D0_X*BR.D0_TO_MU_NUMU_X)
    return fraction_decay*normalisation

def main():
    if len(sys.argv) < 2:
        raise Exception("Not enough arguments.\nPlease specify the number of samples")
    # -------------------------- set up ---------------------------
    beam = BeamExperiment(beta_cm=0.87, gamma_cm=2, s=52000)
    meson = Ds(m=1.87, beam=beam)
    hnl = HNL(m=1.2, beam=beam, parent=meson)
    lepton = Lepton(m=5.11e-4, beam=beam, parent=meson)
    lepton_from_hnl = Lepton(m=5.11e-4, beam=beam, parent=hnl)

    # -------------------------- sampling ---------------------------
    num_samples = int(sys.argv[1])
    start = time.time()
    e0 = (meson.m**2 + hnl.m**2 - lepton.m**2) / (2*meson.m)
    e = np.full(1000, e0)
    cos = np.linspace(0., 1., 1000)
    hnl_rest_samples = ParticleSamples(particle=hnl, decay_dist=lambda e, cos: e/e).generate_samples(e, cos, n_samples=num_samples)

    e_m = np.linspace(meson.m, np.sqrt(beam.s), 10000)
    cos_th_m = np.linspace(0., 1., 10000)
    # Specific non-linear sampling for meson distribution to sample points closer to 1 more finely
    x = -1.0*(cos_th_m - 1)**4 + 1
    # Note the need for the Jacobian factor
    stretched_dist = lambda e, x: (-4*(1-x)**0.75)*meson.distribution(e, x)
    meson_samples = ParticleSamples(particle=meson, decay_dist=stretched_dist).generate_samples(e_m, x, n_samples=num_samples)
    
    # Here we assume the final state leptons are massless 
    # since the differential decay distribution makes this assumption
    # so E+, E- varies from 0 to mN/2
    e_l_plus = np.linspace(0, hnl.m/2, 1000)
    e_l_minus = np.linspace(0, hnl.m/2, 1000)
    lepton_energy_samples = ParticleSamples(particle=lepton_from_hnl, decay_dist=lambda ep, em: lepton_from_hnl.d_gamma_majorana_dEp_dEm(ep, em, DecayType.CCNC)).generate_samples(e_l_plus, e_l_minus, n_samples=num_samples)

    # -------------------------- boost to lab frame ---------------------------
    decay_rate = lepton_from_hnl.total_decay_rate_lepton_pair_from_hnl() 
    # Boost leptons to lab frame
    conversion = 5.08e15 # from metres to GeV^-1
    width_of_detector = DETECTOR_WIDTH * conversion

    hnl_lab_distribution = hnl_rest_samples.boost(meson_samples.samples) \
                                           .geometric_cut(0, np.sqrt(1.022e-3)) \
                                           .add_propagation_factor(width_of_detector, decay_rate)

    di_lepton_samples = get_di_lepton_lab_frame(lepton_energy_samples, hnl_lab_distribution)
    plt.hist2d(di_lepton_samples[:,0], di_lepton_samples[:,1], bins=100, range=[[0,250], [0, 2]])
    plt.show()

    # -------------------------- Cuts and normalise -----------------------------
    BEBC_cut_lepton_samples = apply_BEBC_cuts(di_lepton_samples, e_min=0.8, pT_max=1.85)
    print(f"Decayed HNLs that survive cuts: {len(BEBC_cut_lepton_samples)/len(di_lepton_samples)}")
    total_hnl_decays = get_total_normalised_hnl_decays(BEBC_cut_lepton_samples, num_samples, hnl.m)
    print(f"Mixing^2: {np.sqrt(2/total_hnl_decays)}")
    # ----------------------------------------------------------------------------
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    # --------------------------- write to file -------------------------
    # write_to_file_csv(BEBC_cut_lepton_samples, f"di_lepton_distribution_n={num_samples}.out")

if __name__ == "__main__":
    main()