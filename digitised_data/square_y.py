import sys

def read_csv_file(filename):
    with open(filename) as f:
        lines = f.readlines()

    f = open(f"{filename[:-4]}_sq.csv", "w")
    for line in lines:
        line = line.strip('\n')
        x,y = line.split(',')
        f.write(f"{float(x)},{float(y)**2}\n")
    
read_csv_file(sys.argv[1])