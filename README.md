# HNL Dump
Beam dump simulation for heavy neutral leptons (HNLs)

## Link to paper
Our results can be found in our paper: https://arxiv.org/abs/2208.00416

The final BEBC results that we plot in figures 4 and 5 may be found in `lower_bound_data` and `upper_bound_data`. 

## How to run
The main scripts are `scripts/get_upper_bounds.py` and `scripts/non_linear_calc.py`. Add the `-h` or `--help` when running these scripts to find out how to use them.

Make sure you run the scripts from the top level directory. e.g. `python scripts/get_upper_bounds.py -m 1.2 -n 10000 --mixing tau`

These are used to get the upper and lower bounds of the mixing.
The `non_linear_calc.py` script can be used for both the upper and lower bound and agrees with the results of `get_upper_bounds.py` which uses an approximation to factor out the mixing angle (see https://arxiv.org/abs/2208.00416 eqn. 8-9 for more details).

## Repurposing the code
Majorana/Dirac HNL: The attribute `HNL.majorana` in `src/particles/hnl.py` may be set to True/False, changing the decay rates as appropriate.

Experimental parameters: To find the sensitivity to HNLs of a different experiment, the relevant parameters should be placed in `src/experimental_constants.py`, as well as adding the name of the experiment to `src/experiments.py`. Be aware that currently the MC assumes an on-axis detector.
