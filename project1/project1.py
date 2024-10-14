import sys
import numpy as np
import utils
import genetic_search
import local_search
import logging
import pickle
from bayes_network import BayesNetwork

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
    "small": (1000, True, 0.5, 3, 5), # TODO change n_generations
    "medium": (500, True, 0.5, 2, 5),
}

def compute(infile):
    x, x_header = read_csv_to_array(infile)
    idx2names = {i: x for i, x in enumerate(x_header)}
    infile_name = infile.split('.')[0]
    utils.initLogging(f"both_{infile_name}")

    # n_samples = 10 # 1000
    # local_search.boostrap_fit(x, n_samples)

    population_size, bootstrap_init, \
        structured_init_ratio, max_in_degree, \
        n_generations = param_map[infile_name]


    candidates = genetic_search.compute_genetic_search(x, population_size, bootstrap_init,
        structured_init_ratio, max_in_degree, n_generations)
    
    logging.info("Run local search on best network from genetic search")
    G = candidates[0][0].copy() # highest scoring graph from genetic
    l_search = local_search.StochasticLocalSearch(x, G, max_iter=2500) # TODO tune max_iter
    l_search.fit()
    final_score = l_search.bayesian_score()
    final_edges = l_search.G.edges
    
    logging.info(f"Final score: {final_score}")
    logging.info(f"Final edges: {final_edges}")
    dump_final(l_search.G.copy(), final_score)

    logging.info(f"Size of bayesian score cache: items, {len(BayesNetwork.bayesian_score_cache.keys())}, {sys.getsizeof(BayesNetwork.bayesian_score_cache) / (1024**2)}MB")
    
    outfilename = f"{infile_name}_({round(final_score, 2)}).gph"
    write_gph(l_search.G.copy(), idx2names, outfilename)

def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python project1.py <infile>.csv")

    inputfilename = sys.argv[1]
    compute(inputfilename)


if __name__ == '__main__':
    main()