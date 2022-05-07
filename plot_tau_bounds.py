import os
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from mixing_type import MixingType
from utils import sliced_hist

plt.style.use('ja')

plt.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
def main():
    mixing_type = MixingType.tau
    upper_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}_non_linear.csv")
    lower_bounds = read_csv_file(f"lower_bound_data/lower_bounds_{mixing_type}.csv")
    charm = np.array(read_csv_file("./digitised_data/CHARM_tau.csv"))
    delphi = np.array(read_csv_file("./digitised_data/DELPHI_tau.csv"))
    mathusla = np.array(read_csv_file("./digitised_data/MATHUSLA_tau.csv"))
    na62 = np.array(read_csv_file("./digitised_data/NA62_tau.csv"))
    ship = np.array(read_csv_file("./digitised_data/SHiP_tau.csv"))

    lower_bounds.sort(key=lambda data: data[0], reverse=True)
    upper_bounds.sort(key=lambda data: data[0])
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    bounds = np.append(upper_bounds, lower_bounds, axis=0)

    # # ------------------------ Plotting --------------------------------
    plt.figure(0)
    plt.xlabel(r'$M\, [ \mathrm{GeV}]$')
    plt.ylabel(r'$|U_{\tau}|^2$')
    plt.plot(*get_cols(bounds), 'k')
    # plt.plot(*get_cols(delphi))
    plt.plot(*get_cols(charm))
    # plt.plot(*get_cols(mathusla))
    # plt.plot(*get_cols(na62))
    # plt.plot(*get_cols(ship))
    plt.yscale('log')
    dirname = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dirname, f"recent_graphs/mass_bound_plot[{mixing_type}].png"), dpi=250)
    plt.show()

def get_cols(array: np.ndarray):
    return (array[:,0], array[:,1])

def read_csv_file(filename):
    with open(filename) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip('\n')
        arr = line.split(',')
        arr_f = []
        for el in arr:
            arr_f.append(float(el))
        data.append(arr_f) 
    
    return data


if __name__ == "__main__":
    main()