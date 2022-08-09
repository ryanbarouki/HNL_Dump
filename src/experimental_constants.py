experiment = 2

if experiment == 1:
    DETECTOR_OPENING_ANGLE = 1.69/404 #2.52/404 
    DETECTOR_LENGTH = 1.85 * 5.08e15 # GeV^-1
    DETECTOR_DISTANCE = 404 * 5.08e15 # GeV^-1
    ELECTRON_HNL_CHANNELS = ["e+pos+nu", "mu+e+nu", "e+pi", "mu+mu+nu"]
    TAU_HNL_CHANNELS = ["e+pos+nu", "mu+mu+nu"] 
    ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.72e18
    OBSERVED_EVENTS = 3.5 # 2.3
    BEAM_ENERGY = 400 #GeV
    NUCLEON_MASS = 1. #GeV
    POT = 2.72e18
    
elif experiment == 2:
    DETECTOR_OPENING_ANGLE = 1.69/1400 #2.52/404 
    DETECTOR_LENGTH = 34 * 5.08e15 # GeV^-1
    DETECTOR_DISTANCE = 1400 * 5.08e15 # GeV^-1
    ELECTRON_HNL_CHANNELS = ["e+pos+nu", "mu+e+nu", "e+pi", "mu+mu+nu"]
    TAU_HNL_CHANNELS = ["e+pos+nu", "mu+mu+nu"] 
    ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.72e18
    OBSERVED_EVENTS = 2.3 # 2.3
    BEAM_ENERGY = 800 #GeV
    NUCLEON_MASS = 1. #GeV
    POT = 2.54e18
    
elif experiment == 3:
    DETECTOR_OPENING_ANGLE = 4/50 #2.52/404 
    DETECTOR_LENGTH = 50 * 5.08e15 # GeV^-1
    DETECTOR_DISTANCE = 50 * 5.08e15 # GeV^-1
    ELECTRON_HNL_CHANNELS = ["e+pos+nu", "mu+e+nu", "e+pi", "mu+mu+nu"]
    TAU_HNL_CHANNELS = ["e+pos+nu", "mu+mu+nu"] 
    ELECTRON_NU_MASSLESS_FLUX = 4.1e-4 * 2.72e18
    OBSERVED_EVENTS = 2.3 # 2.3
    BEAM_ENERGY = 800 #GeV
    NUCLEON_MASS = 1. #GeV
    POT = 2.54e18