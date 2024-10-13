import sys
import numpy as np
import networkx as nx
from functools import lru_cache
from scipy.special import loggamma
from itertools import product
import random
from tqdm import tqdm

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
       
def mutual_information(data):
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

# TODO: parrallelize fit, do local search after some population
class GeneticSearch:
    def __init__(self, x, population_size, max_in_degree=4):
        self.x = x
        self.num_nodes = x.shape[1]
        self.max_in_degree = max_in_degree
        self.population_size = population_size
        self.population = []
        self.mi_matrix = mutual_information(x)
    
    def init_population(self, structured_ratio=0.75, bootstrap=True):
        # structured_ratio: number of candidates in initial population
        # generated with mi_ranks
        
        cnt_random_candidates = int((1 - structured_ratio) * self.population_size)
        cnt_structuerd_candidats = self.population_size - cnt_random_candidates

        for _ in tqdm(range(cnt_random_candidates), desc='Init pop, generating random DAGs', disable=True):
            self.population.append(generate_random_dag(self.num_nodes, max_in_degree=2))
        
        for _ in tqdm(range(cnt_structuerd_candidats), desc='Init pop, generating mi ranked random DAGs', disable=True):
            x_ = self.x[np.random.choice(self.x.shape[0], self.x.shape[0], replace=True)] if bootstrap else self.x
            mi_matrix = mutual_information(x_)
            self.population.append(generate_random_dag(self.num_nodes, max_in_degree=2, mi_matrix=mi_matrix))

    def select_candidate(self, candidates):
        fitness_scores = [x[1] for x in candidates]
        population = [x[0] for x in candidates]

        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        return random.choices(population, weights=selection_probs, k=1)[0]
    
    def next_gen(self, mutation_rate=0.025, elite_ratio=0.1):
        # TODO: maybe add some variability into size of next generation
        
        bayesian_networks = [BayesNetwork(self.x, G) for G in self.population]
        candidates = [(b.G.copy(), b.bayesian_score()) for b in bayesian_networks]
        candidates.sort(key = lambda x: x[1], reverse = True)
        
        new_population = []
        elites = [x[0] for x in candidates[:int(self.population_size*elite_ratio)]]
        new_population.extend(elites)

        for _ in tqdm(range(len(elites), self.population_size), desc="Crossover generation", disable=True):
            parent1 = self.select_candidate(candidates)
            parent2 = self.select_candidate(candidates)
            offspring = crossover_bayesian_dags(parent1, parent2, self.max_in_degree, self.mi_matrix)
            offspring = mutate_bayesian_dag(offspring, self.max_in_degree, mutation_rate, self.mi_matrix)
            new_population.append(offspring)
        
        self.population = new_population
        return [BayesNetwork(self.x, G) for G in self.population]


    def fit(self, n_generations):
        print(f'Population size: {self.population_size}')
        for i in tqdm(range(n_generations), desc="Training BN structure", disable=True):
            print(f'Generation: {i+1}')
            bayesian_networks = self.next_gen()

            candidates = [(b.G.copy(), b.bayesian_score()) for b in bayesian_networks]
            candidates.sort(key = lambda x: x[1], reverse = True)

            print(f'\nTop Candidate Bayesian Score: {candidates[0][1]}')
            print(f'Top Candidate edges: {candidates[0][0].edges}')

            print(f'\nWorst candidate Bayesian Score: {candidates[-1][1]}')
            print(f'Worst candidate edges: {candidates[-1][0].edges}')


def compute(infile, outfile):
    x, x_header = read_csv_to_array(infile)

    genetic_search = GeneticSearch(x, population_size=10000, max_in_degree=3)
    genetic_search.init_population()
    genetic_search.fit(n_generations=10)

    print('\nNot bootstrap, higher random init pop')
    genetic_search = GeneticSearch(x, population_size=10000, max_in_degree=3)
    genetic_search.init_population(structured_ratio=.5, bootstrap=False)
    genetic_search.fit(n_generations=10)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()