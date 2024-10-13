import sys
import numpy as np
import networkx as nx
import random
from tqdm import tqdm
import utils
import logging
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from bayes_network import BayesNetwork, mutual_information

    
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

def mutate_bayesian_dag(G, max_in_degree, mutation_rate=0.025, mi_matrix=None):
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
    
    return candidates

def compute_genetic_search(x, population_size, bootstrap_init, structured_init_ratio, max_in_degree, n_generations=10, dumpfilename='dump'):
    logging.info(f"Running GeneticSearch\n population size: {population_size}, boostrap_init: {bootstrap_init}, structured_init_ratio: {structured_init_ratio}, max_in_degree: {max_in_degree}")
    genetic_search = GeneticSearch(x, population_size=population_size, max_in_degree=max_in_degree)
    genetic_search.init_population(structured_ratio=structured_init_ratio, bootstrap=bootstrap_init)
    genetic_search.fit(n_generations=n_generations)
    return dump_last_generation(genetic_search, f"{dumpfilename}") # this returns candidates from last generation
