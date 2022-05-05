from bs4 import BeautifulSoup
import sys

def main():
    if len(sys.argv) < 2:
        raise Exception("Not enough arguments.\n Usage: python xml_to_csv.py <filename>")
    
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, "xml")
    points = bs_data.find_all('point')

    output = open(f"{filename[:-4]}.csv", "w")
    for point in points:
        output.write(f"{point.get('dx')},{point.get('dy')}\n")

if __name__ == "__main__":
    main()