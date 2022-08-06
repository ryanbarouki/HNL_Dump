# HNL Dump
Beam dump simulation for heavy neutral leptons (HNLs)

## Link to paper
Our results can be found in our paper: https://arxiv.org/abs/2208.00416

## How to run
The main scripts are `scripts/get_upper_bounds.py` and `scripts/non_linear_calc.py`. Add the `-h` or `--help` when running these scripts to find out how to use them.

Make sure you run the scripts from the top level directory. e.g. `python scripts/get_upper_bounds.py -m 1.2 -n 10000 --mixing tau`

These are used to get the upper and lower bounds of the mixing.
The `non_linear_calc.py` script can be used for both the upper and lower bound and agrees with the results of `get_upper_bounds.py` which uses an approximation to factor out the mixing angle (see https://arxiv.org/abs/2208.00416 eqn. 8-9 for more details).
