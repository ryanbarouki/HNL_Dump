import numpy as np
import matplotlib.pyplot as plt
from utils import sliced_hist
import sys

def main():
    if len(sys.argv) < 2:
        raise Exception("Not enough arguments.'\n Usage: ./plot <filename>")    

    filename = sys.argv[1] 
    with open(filename) as f:
        lines = f.readlines()

    hnl_lab_distribution = []
    for line in lines:
        line = line.strip('\n')
        arr = line.split(',')
        arr_f = []
        for el in arr:
            arr_f.append(float(el))
        hnl_lab_distribution.append(arr_f) 
    
    hnl_lab_distribution = np.array(hnl_lab_distribution)


    # # ------------------------ Plotting --------------------------------
    plt.figure(0)
    plt.hist2d(hnl_lab_distribution[:,0], hnl_lab_distribution[:,1], bins=100, range=[[0,250], [0.9999, 1]])
    # plt.figure(1)
    # plt.hist(sliced_hist(hnl_lab_distribution, 1, (1-1e-6, 1)), bins=60)
    plt.show()


if __name__ == "__main__":
    main()