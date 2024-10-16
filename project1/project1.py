import sys
import numpy as np
import utils
import genetic_search
import local_search
import logging

def write_gph(dag, idx2names, filename):
    with open(f"graphs/{filename}", 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def read_csv_to_array(infile):
    with open(f'data/{infile}', 'r') as f:
        header = [s.strip('"') for s in f.readline().strip().split(',')]
        x = np.genfromtxt(f, delimiter=',', skip_header=0,dtype='int') # already read line 0
    
    return x, header
 
param_map = {
    "small": (500, True, 0.5, 4, 5), # TODO change n_generations
    # not used
    "medium": (1000, True, 0.5, 3, 10), # will increase max in degree be better?
    "large": (1000, True, 0.5, 3, 5)
}

n_samples_map = {
    "small": 50, # TODO: increase
    "medium": 800,
    "large": 800 # try to increase
}

def compute(infile):
    x, x_header = read_csv_to_array(infile)
    idx2names = {i: x for i, x in enumerate(x_header)}
    infile_name = infile.split('.')[0]
    utils.initLogging(f"both_{infile_name}")

    n_samples = n_samples_map[infile_name]
    k2_networks, l_networks = local_search.boostrap_fit(x, n_samples)
    l_graph = l_networks[0][0]
    k2_graph = k2_networks[0][0]
    
    outfilename_l = f"localS_{infile_name}_({round(l_networks[0][1], 2)}).gph"
    outfilename_k2 = f"K2_{infile_name}_({round(k2_networks[0][1], 2)}).gph"

    write_gph(l_graph.copy(), idx2names, outfilename_l)
    write_gph(k2_graph.copy(), idx2names, outfilename_k2)

    if infile_name in ['medium', 'large']:
        return

    population_size, bootstrap_init, \
        structured_init_ratio, max_in_degree, \
        n_generations = param_map[infile_name]
    

    final_population = genetic_search.compute_genetic_search(x, population_size, bootstrap_init,
        structured_init_ratio, max_in_degree, n_generations)
    
    logging.info("Run local search on best network from genetic search")
    G = final_population[0][0].G.copy() # highest scoring graph from genetic
    outfilename = f"genetic_{infile_name}_({round(final_population[0][1], 2)}).gph"
    write_gph(G.copy(), idx2names, outfilename)

    l_search = local_search.StochasticLocalSearch(x, G, max_iter=2500) # TODO tune max_iter
    l_search.fit()
    final_score = l_search.bayesian_score()
    final_edges = l_search.G.edges
    
    logging.info(f"Final score: {final_score}")
    logging.info(f"Final edges: {final_edges}")
    
    outfilename = f"{infile_name}_({round(final_score, 2)}).gph"
    write_gph(l_search.G.copy(), idx2names, outfilename)

def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python project1.py <infile>.csv")

    inputfilename = sys.argv[1]
    compute(inputfilename)


if __name__ == '__main__':
    main()