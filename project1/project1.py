import sys
import numpy as np
import networkx as nx
import utils
import genetic_search
import local_search
import logging
import pickle

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def read_csv_to_array(infile):
    with open(f'data/{infile}', 'r') as f:
        header = [s.strip('"') for s in f.readline().strip().split(',')]
        x = np.genfromtxt(f, delimiter=',', skip_header=0,dtype='int') # already read line 0
    
    return x, header

def dump_final(G, score):
    with open(f'pickles/final_({(round(score, 2))}).pkl', 'wb') as f:
        pickle.dump(G, f)
    

def compute(infile, outfile):
    x, x_header = read_csv_to_array(infile)
    utils.initLogging(f"both_{infile.split('.')[0]}")

    n_samples = 1000
    local_search.boostrap_fit(x, n_samples)

    population_size = 10000
    bootstrap_init = True
    structured_init_ratio = 0.5
    max_in_degree = 3
    n_generations = 12

    candidates = genetic_search.compute_genetic_search(x, population_size, bootstrap_init,
        structured_init_ratio, max_in_degree, n_generations)
    

    logging.info("Run local search on best network from genetic search")
    G = candidates[0][0].copy() # highest scoring graph from genetic
    l_search = local_search.StochasticLocalSearch(x, G, max_iter=5000) # TODO tune max_iter
    l_search.fit()
    final_score = l_search.bayesian_score()
    final_edges = l_search.G.edges
    

    logging.info(f"Final score: {final_score}")
    logging.info(f"Final edges: {final_edges}")
    dump_final(l_search.G.copy(), final_score)
    
def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()