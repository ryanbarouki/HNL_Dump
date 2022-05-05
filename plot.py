import os
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from mixing_type import MixingType
from utils import sliced_hist
import sys

plt.style.use('ja')

plt.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
def main():
    if len(sys.argv) < 2:
        raise Exception("Not enough arguments.'\n Usage: python plot.py [tau/electron]")    

    mixing_type = MixingType[sys.argv[1]]
    upper_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}.csv")
    # lower_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}.csv")
    lower_bounds = read_csv_file(f"lower_bound_data/lower_bounds_{mixing_type}.csv")
    bebc_data = read_csv_file("./digitised_data/BEBC_bounds_electron.csv")

    lower_bounds.sort(key=lambda data: data[0])
    upper_bounds.sort(key=lambda data: data[0])
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    bebc_data = np.array(bebc_data)

    # # ------------------------ Plotting --------------------------------
    plt.figure(0)
    plt.xlabel(r'$M\, [ \mathrm{GeV}]$')
    y_label = ""
    if mixing_type == MixingType.tau:
        y_label = r'$|U_{\tau}|^2$'
    elif mixing_type == MixingType.electron:
        y_label = r'$|U_{e}|^2$'
        
    plt.ylabel(y_label)
    plt.plot(upper_bounds[:,0], upper_bounds[:,1], 'k')
    plt.plot(lower_bounds[:,0], lower_bounds[:,1], 'k')
    plt.plot(bebc_data[:,0], bebc_data[:,1])
    plt.yscale('log')
    dirname = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dirname, f"recent_graphs/mass_bound_plot[{mixing_type}].png"), dpi=250)
    plt.show()

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