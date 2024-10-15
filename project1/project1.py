import sys
import numpy as np
import utils
import genetic_search
import local_search
import logging
import pickle

"""
# for vectorization
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
"""

def write_gph(dag, idx2names, filename):
    with open(f"graphs/{filename}", 'w') as f:
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
    
param_map = {
    "small": (1000, True, 0.5, 3, 15), # TODO change n_generations
    "medium": (2000, True, 0.5, 3, 20), # will increase max in degree be better?
    "large": (1000, True, 0.5, 3, 20)
}

n_samples_map = {
    "small": 200,
    "medium": 100,
    "large": 50
}

def compute(infile):
    x, x_header = read_csv_to_array(infile)
    idx2names = {i: x for i, x in enumerate(x_header)}
    infile_name = infile.split('.')[0]
    utils.initLogging(f"both_{infile_name}")

    n_samples = n_samples_map[infile_name]
    k2_graphs, l_graphs = local_search.boostrap_fit(x, n_samples)

    population_size, bootstrap_init, \
        structured_init_ratio, max_in_degree, \
        n_generations = param_map[infile_name]
    
    
    candidates = genetic_search.compute_genetic_search(x, population_size, bootstrap_init,
        structured_init_ratio, max_in_degree, n_generations, init_pop=l_graphs[:int(len(l_graphs) * 0.1)]) # add best l_graphs in init pop
    
    logging.info("Run local search on best network from genetic search")
    G = candidates[0][0].copy() # highest scoring graph from genetic
    l_search = local_search.StochasticLocalSearch(x, G, max_iter=2500) # TODO tune max_iter
    l_search.fit()
    final_score = l_search.bayesian_score()
    final_edges = l_search.G.edges
    
    logging.info(f"Final score: {final_score}")
    logging.info(f"Final edges: {final_edges}")
    dump_final(l_search.G.copy(), final_score)

    
    outfilename = f"{infile_name}_({round(final_score, 2)}).gph"
    write_gph(l_search.G.copy(), idx2names, outfilename)

def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python project1.py <infile>.csv")

    inputfilename = sys.argv[1]
    compute(inputfilename)


if __name__ == '__main__':
    main()