import sys
import numpy as np
import utils
import local_search

def write_gph(dag, idx2names, filename):
    with open(f"graphs/{filename}", 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def read_csv_to_array(infile):
    with open(f'data/{infile}', 'r') as f:
        header = [s.strip('"') for s in f.readline().strip().split(',')]
        x = np.genfromtxt(f, delimiter=',', skip_header=0,dtype='int') # already read line 0
    
    return x, header
 

n_samples_map = {
    "small": 10000,
    "medium": 5000,
    "large": 1000
}

def compute(infile):
    x, x_header = read_csv_to_array(infile)
    idx2names = {i: x for i, x in enumerate(x_header)}
    infile_name = infile.split('.')[0]
    utils.initLogging(f"parallel_{infile_name}")

    n_samples = n_samples_map[infile_name]
    k2_networks, l_networks = local_search.boostrap_fit(x, n_samples)
    l_graph = l_networks[0][0]
    k2_graph = k2_networks[0][0]
    
    outfilename_l = f"parallel_local_{infile_name}_({round(l_networks[0][1], 2)}).gph"
    outfilename_k2 = f"parallel_k2_{infile_name}_({round(k2_networks[0][1], 2)}).gph"

    write_gph(l_graph.copy(), idx2names, outfilename_l)
    write_gph(k2_graph.copy(), idx2names, outfilename_k2)

def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python project1.py <infile>.csv")

    inputfilename = sys.argv[1]
    compute(inputfilename)

if __name__ == '__main__':
    main()