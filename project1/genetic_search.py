import sys
import numpy as np
import networkx as nx
from functools import lru_cache
from scipy.special import loggamma
from itertools import product
import random
from tqdm import tqdm
import utils
import logging
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def read_csv_to_array(infile):
    with open(f'data/{infile}', 'r') as f:
        header = [s.strip('"') for s in f.readline().strip().split(',')]
        x = np.genfromtxt(f, delimiter=',', skip_header=0,dtype='int') # already read line 0
    
    return x, header
    
class BayesNetwork:
    def __init__(self, x, G=None):
        self.x = x
        self.x_values_range = np.max(x, axis=0)
        self.n = x.shape[1]
        self.n_obs = x.shape[0]

        if G == None:
            self.G = nx.DiGraph()
            self.G.add_nodes_from(range(self.n))
        else:
            self.G = G

        self.value_index_map = {} # node: {value: ndarray(bool)} # True if x[:, index] == value
        for i in range(self.n):
            self.value_index_map[i] = {}
            for j in range(1, self.x_values_range[i] + 1):
                self.value_index_map[i][j] = (self.x[:, i] == j)
    
    def copy(self):
        new_bn = BayesNetwork(self.x)
        new_bn.G = self.G.copy()
        new_bn.value_index_map = {k: v.copy() for k, v in self.value_index_map.items()}
        return new_bn

    # TODO: will it become necessary to encode/enumerate j
    @lru_cache(maxsize=None)
    def m(self, i, j):
        # node i
        # j, enumeration of parent node instantiation. represented with tuple of tuples ((node, val))         
        idx = np.full(self.n_obs, True)
        if j:
            for parent_node, parent_node_val in j:
                idx = np.logical_and(idx, self.value_index_map[parent_node][parent_node_val])

        i_vals = (self.x[idx])[:,i]
        m_ijk_ = np.array([np.sum(i_vals == k) for k in range(1, self.x_values_range[i] + 1)])
        m_ij0 = np.sum(idx)
        return m_ijk_, m_ij0
    
    @lru_cache(maxsize=None)
    def get_parent_instantiations(self, parents):        
        parents_vals = [list(range(1, self.x_values_range[parent] + 1)) for parent in parents]
        product_vals = product(*parents_vals)
        return [tuple(zip(parents, values)) for values in product_vals]
    
    def bayesian_score(self):
        p = 0
        for i in range(self.n):
            parents = tuple(self.G.predecessors(i))
                        
            q = self.get_parent_instantiations(parents) if parents else [()] # case where node has no parents
            r = self.x_values_range[i]
            for j in q:
                m_ijk, m_ij0 = self.m(i, j)
                alpha_ij0 = r # uniform prior, each alpha_ijk has value 1
                p += (loggamma(alpha_ij0) - loggamma(alpha_ij0 + m_ij0))
                for k in range(r): # shifted for 0-indexing
                    p += loggamma(1 + m_ijk[k]) # uniform prior, denominator term eliminated
        
        return p

# somewhat modified mutual_information       
def mutual_information(data, constraints=None):
    # constraints: {node: [lowest(list of forced lowest mi rank in order), highest()]}

    bayes_net = BayesNetwork(data)
    x_values_range = bayes_net.x_values_range
    n = bayes_net.n
    n_obs = bayes_net.n_obs

    value_index_map = bayes_net.value_index_map
    mi_matrix = np.zeros((n, n))
    
    def pairwise_mi(i, j):
        mi = 0
        for x_i in range(1, x_values_range[i] + 1):
            idx_i = value_index_map[i][x_i]
            p_xi = np.count_nonzero(idx_i) / n_obs
            for x_j in range(1, x_values_range[j] + 1):
                idx_j = value_index_map[j][x_j]
                p_xj = np.count_nonzero(idx_j) / n_obs
                idx_joint = np.logical_and(idx_i, idx_j)
                p_joint = np.count_nonzero(idx_joint) / n_obs
                if p_joint > 0:
                    mi += p_joint * (np.log(p_joint) - (np.log(p_xi) + np.log(p_xj)))
        return mi

    for i in range(n):
        for j in range(i+1, n):
            mi_matrix[i, j] = mi_matrix[j, i] = pairwise_mi(i, j)
    
    if constraints:
        mi_max_copy = mi_matrix.copy()
        mi_min_copy = mi_matrix.copy()
        np.fill_diagonal(mi_max_copy, -np.inf)
        np.fill_diagonal(mi_min_copy, np.inf)
        for i in range(n):
            if i in constraints.keys():
                std = np.std(mi_matrix[i, :]) # use this or some other

                # note this makes the mi_matrix unsymmetric, but it should be like that
                if 'h' in constraints[i].keys():
                    max_i = np.max(mi_max_copy[i, :]) + std
                    for n in reversed(constraints[i]['h']):
                        mi_matrix[i][n] = max_i
                        max_i += std
                
                if 'l' in constraints[i].keys():
                    min_i = np.min(mi_min_copy[i, :]) - std
                    for n in reversed(constraints[i]['l']):
                        mi_matrix[i][n] = min_i
                        min_i -= std
                    
    return mi_matrix

def generate_random_dag(num_nodes, max_in_degree=2, mi_matrix=None):
    # generate random DAG
    # if mi_matrix: then use mutual_information 
    # score to weigh parent node probabilities
    G = nx.DiGraph()
    nodes = list(range(num_nodes))
    G.add_nodes_from(nodes)

    random.shuffle(nodes)
    for n in nodes:
        parent_pool = [x for x in nodes if x != n]

        if mi_matrix is not None:
            mi_scores = np.array([mi_matrix[x][n] for x in parent_pool])
            total_mi = np.sum(mi_scores)
        
            if total_mi == 0:
                probs = np.ones(len(parent_pool)) / len(parent_pool)
            else:
                probs = mi_scores / total_mi
        else:
            probs = np.ones(len(parent_pool)) / len(parent_pool)
        
        num_parents = random.randint(0, min(len(parent_pool), max_in_degree))
        if num_parents > 0:
            selected_parents = np.random.choice(
                parent_pool,
                size=num_parents,
                replace=False,
                p=probs
            )
            for parent in selected_parents:
                # keep acyclic
                if not nx.has_path(G, n, parent):
                    G.add_edge(parent, n)
    
    return G


def crossover_bayesian_dags(parent1, parent2, max_in_degree, mi_matrix=None):
    offspring = nx.DiGraph()
    offspring.add_nodes_from(parent1.nodes())
    nodes = list(offspring.nodes())
    random.shuffle(nodes)

    for node in nodes:
        combined_parent_nodes = list(set(parent1.predecessors(node)).union(set(parent2.predecessors(node))))

        if len(combined_parent_nodes) > max_in_degree:
            if mi_matrix is not None:
                mi_scores = {pred: mi_matrix[pred][node] for pred in combined_parent_nodes}
                sorted_preds = sorted(mi_scores.items(), key=lambda item: item[1], reverse=True)
                selected_preds = [pred for pred, _ in sorted_preds[:max_in_degree]]
            else:
                random.shuffle(combined_parent_nodes)
                selected_preds = combined_parent_nodes[:max_in_degree]
        else:
            selected_preds = combined_parent_nodes
        
        for pred in selected_preds:
            if not nx.has_path(offspring, node, pred):
                offspring.add_edge(pred, node)
    
    return offspring

def mutate_bayesian_dag(G, max_in_degree, mutation_rate=0.001, mi_matrix=None):
    # TODO: tune mutation_rate
    mutated_G = G.copy()
    nodes = list(mutated_G.nodes())
    random.shuffle(nodes) # should not matter

    # Edge Deletion
    for edge in list(mutated_G.edges()):
        if random.random() < (mutation_rate/len(nodes)):
            mutated_G.remove_edge(*edge)

    # Edge reversal
    for edge in list(mutated_G.edges()):
        if random.random() < (mutation_rate/len(nodes)):
            u, v = edge
            if mutated_G.in_degree(u) < max_in_degree and not nx.has_path(mutated_G, u, v):
                mutated_G.remove_edge(u, v)
                mutated_G.add_edge(v, u)
    
    # Edge addition
    if random.random() < mutation_rate:
        u, v = random.sample(nodes, 2)
        if u != v and not mutated_G.has_edge(u, v) and not mutated_G.has_edge(v, u):
            if mutated_G.in_degree(v) < max_in_degree and not nx.has_path(mutated_G, v, u):
                mutated_G.add_edge(u, v)
    
    return mutated_G

def create_bayes_network(args):
    x, G = args
    return BayesNetwork(x, G)

def generate_offspring(args):
    parent1, parent2, max_in_degree, mi_matrix, mutation_rate = args
    offspring = crossover_bayesian_dags(parent1, parent2, max_in_degree, mi_matrix)
    mutated_offspring = mutate_bayesian_dag(offspring, max_in_degree, mutation_rate, mi_matrix)
    return mutated_offspring

def compute_candidate(bayes_net):
    return (bayes_net.G.copy(), bayes_net.bayesian_score())

def generate_random_dag_wrapper(args):
    num_nodes, max_in_degree, mi_matrix = args
    return generate_random_dag(num_nodes, max_in_degree, mi_matrix)

def generate_structured_random_dag(args):
    x, num_nodes, max_in_degree, bootstrap = args
    x_ = x[np.random.choice(x.shape[0], x.shape[0], replace=True)] if bootstrap else x
    mi_matrix = mutual_information(x_)
    return generate_random_dag(num_nodes, max_in_degree, mi_matrix=mi_matrix)

class GeneticSearch:
    def __init__(self, x, population_size, max_in_degree=4, mi_constraints=None):
        self.x = x
        self.num_nodes = x.shape[1]
        self.max_in_degree = max_in_degree
        self.population_size = population_size
        self.population = []
        self.mi_matrix = mutual_information(x, mi_constraints)
    
    def init_population(self, structured_ratio=0.75, bootstrap=True):
        # structured_ratio: fraction of candidates generated with mutual information ranking
        logging.info(f"Generating initial population with structured ratio: {structured_ratio}")
        cnt_random_candidates = int((1 - structured_ratio) * self.population_size)
        cnt_structured_candidates = self.population_size - cnt_random_candidates

        num_cores = multiprocessing.cpu_count()
        
        # Generate random DAGs without mi_matrix in parallel
        args_list = [(self.num_nodes, 2, None) for _ in range(cnt_random_candidates)]
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            random_dags = list(executor.map(generate_random_dag_wrapper, args_list))
        
        self.population.extend(random_dags)
        
        # Generate DAGs with mi_matrix in parallel
        args_list = [(self.x, self.num_nodes, 2, bootstrap) for _ in range(cnt_structured_candidates)]
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            structured_dags = list(executor.map(generate_structured_random_dag, args_list))
        
        self.population.extend(structured_dags)

    def select_candidate(self, candidates):
        fitness_scores = [x[1] for x in candidates]
        population = [x[0] for x in candidates]

        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        return random.choices(population, weights=selection_probs, k=1)[0]
    
    def next_gen(self, mutation_rate=0.025, elite_ratio=0.1):
        num_cores = multiprocessing.cpu_count()

        args_list = [(self.x, G) for G in self.population]
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            bayesian_networks = list(executor.map(create_bayes_network, args_list))

        # Compute candidates
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            candidates = list(executor.map(compute_candidate, bayesian_networks))        

        candidates.sort(key=lambda x: x[1], reverse=True)

        elite_count = int(self.population_size * elite_ratio)
        elites = [x[0] for x in candidates[:elite_count]]
        new_population = elites.copy()

        # Prepare arguments for generate_offspring
        offspring_count = self.population_size - elite_count
        args_list = [
            (
                self.select_candidate(candidates),
                self.select_candidate(candidates),
                self.max_in_degree,
                self.mi_matrix,
                mutation_rate,
            )
            for _ in range(offspring_count)
        ]

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            offspring_list = list(executor.map(generate_offspring, args_list))

        new_population.extend(offspring_list)
        self.population = new_population

        # Prepare arguments for create_bayes_network again
        args_list = [(self.x, G) for G in self.population]
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            return list(executor.map(create_bayes_network, args_list))


    def fit(self, n_generations):
        num_cores = multiprocessing.cpu_count()
        logging.info(f'Start training GeneticSearch')
        for i in tqdm(range(n_generations), desc="Training BN structure", disable=True):
            logging.info(f'Generation: {i+1}')
            bayesian_networks = self.next_gen()

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                candidates = list(executor.map(compute_candidate, bayesian_networks))

            candidates = [(b.G.copy(), b.bayesian_score()) for b in bayesian_networks]
            candidates.sort(key = lambda x: x[1], reverse = True)

            logging.info(f'Top Candidate Bayesian Score: {candidates[0][1]}')
            logging.info(f'Top Candidate edges: {candidates[0][0].edges}')

            logging.info(f'Worst candidate Bayesian Score: {candidates[-1][1]}')
            logging.info(f'Worst candidate edges: {candidates[-1][0].edges}')
        
        logging.info(f'Traning GeneticSearch completed')

def dump_last_generation(genetic_search: GeneticSearch, filename):
    num_cores = multiprocessing.cpu_count()

    args_list = [(genetic_search.x, G) for G in genetic_search.population]
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        bayesian_networks = list(executor.map(create_bayes_network, args_list))
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        candidates = list(executor.map(compute_candidate, bayesian_networks))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_score = candidates[0][1]

    with open(f'pickles/{filename}_({(round(top_score, 2))}).pkl', 'wb') as f:
        pickle.dump(candidates, f)

def compute(infile, outfile):
    x, x_header = read_csv_to_array(infile)
    utils.initLogging(f"genetic_{infile.split('.')[0]}")

    n = x.shape[1]
    mi_constraints = {i:{'l': [n-1]} for i in range(n - 1)} # force "response" variable to have lowest mi_rank
    population_size = 25000

    logging.info(f"Running GeneticSearch, population size {population_size}, default params")
    genetic_search = GeneticSearch(x, population_size=population_size, max_in_degree=3)
    genetic_search.init_population()
    genetic_search.fit(n_generations=10)
    dump_last_generation(genetic_search, f"{infile.split('.')[0]}_default")

    logging.info(f"Running GeneticSearch, population size {population_size}, No bootstap, majority random initial population")
    genetic_search = GeneticSearch(x, population_size=population_size, max_in_degree=3)
    genetic_search.init_population(structured_ratio=.25, bootstrap=False)
    genetic_search.fit(n_generations=10)
    dump_last_generation(genetic_search, f"{infile.split('.')[0]}_more_random")

    logging.info(f"Running GeneticSearch, population size {population_size}, default params, with mi_constraints")
    genetic_search = GeneticSearch(x, population_size=population_size, max_in_degree=3, mi_constraints=mi_constraints)
    genetic_search.init_population()
    genetic_search.fit(n_generations=10)
    dump_last_generation(genetic_search, f"{infile.split('.')[0]}_default")

    logging.info(f"Running GeneticSearch, population size {population_size}, No bootstap, majority random initial population, with mi_constraints")
    genetic_search = GeneticSearch(x, population_size=population_size, max_in_degree=3, mi_constraints=mi_constraints)
    genetic_search.init_population(structured_ratio=.25, bootstrap=False)
    genetic_search.fit(n_generations=10)
    dump_last_generation(genetic_search, f"{infile.split('.')[0]}_more_random")


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()