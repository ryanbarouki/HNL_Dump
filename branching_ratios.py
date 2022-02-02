import numpy as np
from particle_masses import *
D_TO_E_NUE_X = 16.07/100 # +/- 0.30%
D_TO_MU_NUMU_X = 17.6/100 # +/- 3.2%

def D_TO_MU_HNL(mN):
    mD = D_MASS # GeV
    ml = MUON_MASS # GeV
    x = (ml/mD)**2
    y = (mN/mD)**2
    l = lambda x, y: 1 + x**2 + y**2 - 2*x*y - 2*x - 2*y
    h = lambda x, y: x + y - (x-y)**2
    l_tilde = l(x,y)/l(x,0)
    h_tilde = h(x,y)/h(x,0)
    # BR_D_TO_E_NUE = 8.3e-5
    BR_D_TO_MU_NUMU = 3.74e-4 # +/- 0.17e-4
    return BR_D_TO_MU_NUMU*(l_tilde**0.5)*h_tilde

def D_TO_E_HNL(mN):
    mD = D_MASS # GeV
    ml = ELECTRON_MASS # GeV
    x = (ml/mD)**2
    y = (mN/mD)**2
    l = lambda x, y: 1 + x**2 + y**2 - 2*x*y - 2*x - 2*y
    h = lambda x, y: x + y - (x-y)**2
    l_tilde = l(x,y)/l(x,0)
    h_tilde = h(x,y)/h(x,0)
    BR_D_TO_E_NUE = 1.5e-8
    return BR_D_TO_E_NUE*np.sqrt(l_tilde)*h_tilde

D0_TO_E_NUE_X = 6.49/100 # +/- 0.11%
D0_TO_MU_NUMU_X = 6.8/100 # +/- 0.6%
