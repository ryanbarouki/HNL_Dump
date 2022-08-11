# --- Needed to tell python about the package in ../src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))
# ---
import os
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from src.mixing_type import MixingType
from src.utils import sliced_hist

plt.style.use('ja')

plt.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
def main():
    mixing_type = MixingType.electron
    upper_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}.csv")
    lower_bounds = read_csv_file(f"lower_bound_data/lower_bounds_{mixing_type}_non_linear.csv")
    bebc = np.array(read_csv_file("digitised_data/electron/BEBC.csv"))
    charm = np.array(read_csv_file("digitised_data/electron/CHARM.csv"))
    belle = np.array(read_csv_file("digitised_data/electron/Belle.csv"))
    delphi = np.array(read_csv_file("digitised_data/electron/DELPHI.csv"))
    t2k = np.array(read_csv_file("digitised_data/electron/T2K.csv"))
    dune = np.array(read_csv_file("digitised_data/electron/DUNE.csv"))
    mathusla = np.array(read_csv_file("digitised_data/electron/MATHUSLA.csv"))
    na62 = np.array(read_csv_file("digitised_data/electron/NA62.csv"))
    ship = np.array(read_csv_file("digitised_data/electron/SHiP.csv"))
    x,y = np.array(getCoords("digitised_data/electron/HNLeCHARM.dat"))

    lower_bounds.sort(key=lambda data: data[0], reverse=True)
    upper_bounds.sort(key=lambda data: data[0])
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    bounds = np.append(upper_bounds, lower_bounds, axis=0)
    
    # x0, y0 = x.mean(), y.mean()
    # angle = np.arctan2(y - y0, x - x0)
    
    # idx = angle.argsort()
    # x, y = x[idx], y[idx]


    # # ------------------------ Plotting --------------------------------
    dpi = 250
    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    plt.xlabel(r'$M_N\, [ \mathrm{GeV}]$')
    plt.ylabel(r'$|U_{eN}|^2$')
    plt.plot(*get_cols(bounds), 'k', label="BEBC (reanalysis)")
    plt.plot(x,y, label="CHARM (recast)")
    plt.plot(*get_cols(t2k), label="T2K")
    plt.plot(*get_cols(belle), label="Belle (recast)")
    plt.plot(*get_cols(delphi), label="DELPHI")
    # plt.plot(*get_cols(na62), '--', dashes=(5,2), label="NA62")
    # plt.plot(*get_cols(dune), '--', dashes=(5,2), label="DUNE")
    # plt.plot(*get_cols(mathusla), '--', dashes=(5,2), label="MATHUSLA")
    # plt.plot(*get_cols(ship), '--', dashes=(5,2), label="SHiP")
    plt.legend(fontsize="small")
    plt.yscale('log')
    plt.xlim([0.1,3])
    plt.fill(*get_cols(bounds), color="#ddd")
    plt.fill_between(*get_cols(delphi), 1, color="#ddd")
    plt.fill_between(*get_cols(belle), 1, color="#ddd")
    plt.fill_between(*get_cols(t2k), 1, color="#ddd")
    plt.fill_between(x,y, 1, color="#ddd")
    dirname = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dirname, f"../recent_graphs/mass_bound_plot{mixing_type}.png"), dpi=dpi)
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

def getCoords(fileName, rescaleX= 1, rescaleY = 1):
    # fullPath = os.path.join(dirname,fileName)
    experimentArray = np.loadtxt(fileName)
    experimentX = np.transpose(experimentArray)[0]*rescaleX
    experimentY = np.transpose(experimentArray)[1]*rescaleY
    return experimentX, experimentY

if __name__ == "__main__":
    main()
