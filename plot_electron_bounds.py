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
    mixing_type = MixingType.electron
    upper_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}.csv")
    # lower_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}.csv")
    lower_bounds = read_csv_file(f"lower_bound_data/lower_bounds_{mixing_type}.csv")
    bebc = np.array(read_csv_file("./digitised_data/BEBC_electron.csv"))
    dune = np.array(read_csv_file("./digitised_data/DUNE_electron.csv"))
    mathusla = np.array(read_csv_file("./digitised_data/MATHUSLA_electron.csv"))
    na62 = np.array(read_csv_file("./digitised_data/NA62_electron.csv"))
    ship = np.array(read_csv_file("./digitised_data/SHiP_electron.csv"))

    lower_bounds.sort(key=lambda data: data[0], reverse=True)
    upper_bounds.sort(key=lambda data: data[0])
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    bounds = np.append(upper_bounds, lower_bounds, axis=0)

    # # ------------------------ Plotting --------------------------------
    plt.figure(0)
    plt.xlabel(r'$M\, [ \mathrm{GeV}]$')
    plt.ylabel(r'$|U_{e}|^2$')
    plt.plot(*get_cols(bebc), label="Old BEBC")
    plt.plot(*get_cols(na62), label="NA62")
    plt.plot(*get_cols(dune), '--', dashes=(5,2), label="DUNE")
    plt.plot(*get_cols(mathusla), '--', dashes=(5,2), label="MATHUSLA")
    plt.plot(*get_cols(ship), '--', dashes=(5,2), label="SHiP")
    plt.plot(*get_cols(bounds), 'k', label="Our results")
    plt.legend(fontsize="small")
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