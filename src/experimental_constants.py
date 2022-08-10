from src.experiment import Experiment

def get_experimental_constants(experiment):

    if experiment == Experiment.BEBC: 
        DETECTOR_OPENING_ANGLE = 1.69/404 #2.52/404 
        DETECTOR_LENGTH = 1.85 * 5.08e15 # GeV^-1
        DETECTOR_DISTANCE = 404 * 5.08e15 # GeV^-1
        ELECTRON_HNL_CHANNELS = ["e+pos+nu", "mu+e+nu", "e+pi", "mu+mu+nu"]
        TAU_HNL_CHANNELS = ["e+pos+nu", "mu+mu+nu"] 
        ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.72e18
        OBSERVED_EVENTS = 2.3 # 2.3
        BEAM_ENERGY = 400 #GeV
        NUCLEON_MASS = 1. #GeV
        POT = 2.72e18
        bMesonFraction = 1.6e-7
        muon_e_min = 3. #GeV
        mT_max = 1.85 #GeV
        B_MESON_FRACTION = 1.6e-7
        
    elif experiment == Experiment.NuTeV:
        DETECTOR_OPENING_ANGLE = 1.69/1400 #2.52/404 
        DETECTOR_LENGTH = 34 * 5.08e15 # GeV^-1
        DETECTOR_DISTANCE = 1400 * 5.08e15 # GeV^-1
        ELECTRON_HNL_CHANNELS = ["mu+mu+nu"]
        TAU_HNL_CHANNELS = ["mu+mu+nu"] 
        ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.54e18
        OBSERVED_EVENTS = 2.3 # 2.3
        BEAM_ENERGY = 800 #GeV
        NUCLEON_MASS = 1. #GeV
        POT = 2.54e18
        B_MESON_FRACTION = 1.6e-6
        
    elif experiment == Experiment.SHiP: # SHiP
        DETECTOR_OPENING_ANGLE = 4/50 #2.52/404 
        DETECTOR_LENGTH = 50 * 5.08e15 # GeV^-1
        DETECTOR_DISTANCE = 50 * 5.08e15 # GeV^-1
        ELECTRON_HNL_CHANNELS = ["e+pos+nu", "mu+e+nu", "e+pi", "mu+mu+nu"]
        TAU_HNL_CHANNELS = ["e+pos+nu", "mu+mu+nu"] 
        ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.72e20
        OBSERVED_EVENTS = 2.3 # 2.3
        BEAM_ENERGY = 400 #GeV
        NUCLEON_MASS = 1. #GeV
        POT = 2.54e20
        bMesonFraction = 1.6e-7
        muon_e_min = 3. #GeV
        mT_max = 1.85 #GeV

            

    return (DETECTOR_OPENING_ANGLE, DETECTOR_LENGTH, DETECTOR_DISTANCE, 
            ELECTRON_HNL_CHANNELS, TAU_HNL_CHANNELS, 
            ELECTRON_NU_MASSLESS_FLUX,OBSERVED_EVENTS,
            BEAM_ENERGY,NUCLEON_MASS,POT, B_MESON_FRACTION, muon_e_min, mT_max)
