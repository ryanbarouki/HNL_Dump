from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from mixing_type import MixingType
from utils import sliced_hist
import sys

def main():
    if len(sys.argv) < 2:
        raise Exception("Not enough arguments.'\n Usage: python plot.py [tau/electron]")    

    mixing_type = MixingType[sys.argv[1]]
    upper_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}_non_linear.csv")
    # lower_bounds = read_csv_file(f"upper_bound_data/upper_bounds_{mixing_type}.csv")
    lower_bounds = read_csv_file(f"lower_bound_data/lower_bounds_{mixing_type}.csv")

    lower_bounds.sort(key=lambda data: data[0])
    upper_bounds.sort(key=lambda data: data[0])
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # # ------------------------ Plotting --------------------------------
    plt.figure(0)
    plt.plot(upper_bounds[:,0], upper_bounds[:,1], 'k')
    plt.plot(lower_bounds[:,0], lower_bounds[:,1], 'k')
    plt.yscale('log')
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