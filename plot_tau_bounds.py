from functools import update_wrapper
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
    upper_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}.csv")
    upper_bounds = [upper_bounds[i] for i in range(len(upper_bounds)) if i % 5 == 0]
    lower_bounds = read_csv_file(f"lower_bound_data/lower_bounds_{mixing_type}_non_linear.csv")
    charm = np.array(read_csv_file("./digitised_data/tau/CHARM.csv"))
    t2k = np.array(read_csv_file("./digitised_data/tau/T2K.csv"))
    delphi = np.array(read_csv_file("./digitised_data/tau/DELPHI.csv"))
    mathusla = np.array(read_csv_file("./digitised_data/tau/MATHUSLA.csv"))
    na62 = np.array(read_csv_file("./digitised_data/tau/NA62.csv"))
    ship = np.array(read_csv_file("./digitised_data/tau/SHiP.csv"))

    lower_bounds.sort(key=lambda data: data[0], reverse=True)
    upper_bounds.sort(key=lambda data: data[0])
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    bounds = np.append(upper_bounds, lower_bounds, axis=0)

    # # ------------------------ Plotting --------------------------------
    dpi = 250
    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    plt.xlabel(r'$M\, [ \mathrm{GeV}]$')
    plt.ylabel(r'$|U_{\tau}|^2$')
    plt.plot(*get_cols(bounds), 'k', label="Our results")
    plt.plot(*get_cols(delphi), label="DELPHI")
    plt.plot(*get_cols(charm), label="CHARM")
    plt.plot(*get_cols(t2k), label="T2K")
    plt.plot(*get_cols(na62), '--', dashes=(5,2), label="NA62")
    plt.plot(*get_cols(ship), '--', dashes=(5,2), label="SHiP")
    plt.plot(*get_cols(mathusla), '--', dashes=(5,2), label="MATHUSLA")
    plt.yscale('log')
    plt.legend(fontsize="small")
    plt.xlim([min(charm[:,0]),3])
    dirname = os.path.dirname(os.path.abspath(__file__))
    plt.fill(*get_cols(bounds), color="#ddd")
    # plt.fill(*get_cols(delphi), color="#ddd")
    plt.fill_between(*get_cols(delphi), 1, color="#ddd")
    # plt.fill(*get_cols(charm), color="#ddd")
    plt.fill_between(*get_cols(charm), 1, color="#ddd")
    plt.fill_between(*get_cols(t2k), 1, color="#ddd")
    plt.savefig(os.path.join(dirname, f"recent_graphs/mass_bound_plot[{mixing_type}].png"), dpi=dpi)
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