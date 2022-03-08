import numpy as np
from particle_masses import *

D_TO_E_NUE_X = 16.07/100 # +/- 0.30%
D_TO_MU_NUMU_X = 17.6/100 # +/- 3.2%
D0_TO_E_NUE_X = 6.49/100 # +/- 0.11%
D0_TO_MU_NUMU_X = 6.8/100 # +/- 0.6%
DS_TO_TAU_X = 1 # TODO find value PDG

def l_tilde(x, y):
    l = lambda x, y: 1 + x**2 + y**2 - 2*x*y - 2*x - 2*y
    return l(x,y)/l(x,0)

def h_tilde(x, y):
    h = lambda x, y: x + y - (x-y)**2
    return h(x,y)/h(x,0)

def D_TO_MU_HNL(mN):
    mD = D_MASS # GeV
    ml = MUON_MASS # GeV
    x = (ml/mD)**2
    y = (mN/mD)**2
    l_til = l_tilde(x, y)
    h_til = h_tilde(x, y)
    # BR_D_TO_E_NUE = 8.3e-5
    BR_D_TO_MU_NUMU = 3.74e-4 # +/- 0.17e-4
    return BR_D_TO_MU_NUMU*np.sqrt(l_til)*h_til

def D_TO_E_HNL(mN):
    x = (ELECTRON_MASS/D_MASS)**2
    y = (mN/D_MASS)**2
    l_til = l_tilde(x, y)
    h_til = h_tilde(x, y)
    BR_D_TO_E_NUE = 1.5e-8
    return BR_D_TO_E_NUE*np.sqrt(l_til)*h_til

def DS_TO_TAU_HNL(mN):
    x = (TAU_MASS/DS_MASS)**2
    y = (mN/DS_MASS)**2
    l_til = l_tilde(x, y)
    h_til = h_tilde(x, y)
    DS_TO_TAU_NUTAU = 1 # TODO find value from PDG
    return DS_TO_TAU_NUTAU*np.sqrt(l_til)*h_til

def TAU_TO_PI_HNL(mN):
    x = (PION_MASS/TAU_MASS)**2
    y = (mN/TAU_MASS)**2
    l_til = l_tilde(x, y)
    h_til = h_tilde(x, y)
    TAU_TO_PI_NUTAU = 1 # TODO find value from PDG
    return TAU_TO_PI_NUTAU*np.sqrt(l_til)*h_til