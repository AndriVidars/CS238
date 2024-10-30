import sys
import numpy as np
import networkx as nx
import random
import logging
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from bayes_network import BayesNetwork, mutual_information
from typing import List

# TODO tune max in degree
def generate_random_bayes_net(x, num_nodes, max_in_degree=4, mi_matrix=None):
    # generate random DAG
    # if mi_matrix: then use mutual_information score to weigh parent node probabilities

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

    bn = BayesNetwork(x, G)
    y = bn.bayesian_score()    
    return bn, y


def crossover_bayesian_networks(parent1: BayesNetwork, parent2: BayesNetwork, y1, max_in_degree, mi_matrix=None):
    offspring = parent1.copy()
    y_offspring = y1
    nodes = list(offspring.G.nodes())
    random.shuffle(nodes)

    for node in nodes:
        combined_parent_nodes = list(set(parent1.G.predecessors(node)).union(set(parent2.G.predecessors(node))))

        if len(combined_parent_nodes) > max_in_degree:
            if mi_matrix is not None:
                mi_scores = {pred: mi_matrix[pred][node] for pred in combined_parent_nodes}
                sorted_preds = sorted(mi_scores.items(), key=lambda item: item[1], reverse=True) # what was I doing here ?
                selected_preds = [pred for pred, _ in sorted_preds[:max_in_degree]]
            else:
                random.shuffle(combined_parent_nodes)
                selected_preds = combined_parent_nodes[:max_in_degree]
        else:
            selected_preds = combined_parent_nodes
        
        for pred in selected_preds:
            if not nx.has_path(offspring.G, node, pred):
                i, j = pred, node
                parents_curr = list(offspring.G.predecessors(j))
                parents_next = parents_curr + [i]
                delta_y = offspring.bayesian_score_delta(j, tuple(sorted(parents_curr)), tuple(sorted(parents_next)))
                y_offspring += delta_y
                offspring.G.add_edge(i, j)
    
    return offspring, y_offspring

 # TODO: tune mutation_rate
def mutate_bayesian_network(bayes_net: BayesNetwork, y, max_in_degree, mutation_rate=0.1, mi_matrix=None):
    nodes = list(bayes_net.G.nodes())
    # Edge Deletion
    for edge in list(bayes_net.G.edges()):
        if random.random() < (2*mutation_rate/(len(nodes))): # give higher probability to removing node
            i, j = edge
            parents_curr = list(bayes_net.G.predecessors(j))
            parents_next = [x for x in parents_curr if x != i]
            delta_y = bayes_net.bayesian_score_delta(j, tuple(sorted(parents_curr)), tuple(sorted(parents_next)))
            bayes_net.G.remove_edge(i, j)
            y += delta_y

    # Edge addition
    if random.random() < mutation_rate:
        i, j = random.sample(nodes, 2)
        if i != j and not bayes_net.G.has_edge(i, j) and not bayes_net.G.has_edge(j, i):
            if bayes_net.G.in_degree(j) < max_in_degree and not nx.has_path(bayes_net.G, j, i):
                parents_curr = list(bayes_net.G.predecessors(j))
                parents_next = parents_curr + [i]
                delta_y = bayes_net.bayesian_score_delta(j, tuple(sorted(parents_curr)), tuple(sorted(parents_next)))
                bayes_net.G.add_edge(i, j)
                y += delta_y
    
    return bayes_net, y


def generate_offspring(args):
    candidate1, candidate2, max_in_degree, mi_matrix, mutation_rate = args
    parent1, parent2 = candidate1[0], candidate2[0]
    y1 = candidate1[1]

    offspring, y_offspring = crossover_bayesian_networks(parent1, parent2, y1, max_in_degree, mi_matrix)
    mutated_offspring, y_offspring = mutate_bayesian_network(offspring, y_offspring, max_in_degree, mutation_rate, mi_matrix)
    return mutated_offspring, y_offspring


def generate_random_bayes_net_wrapper(args):
    x, num_nodes, max_in_degree, mi_matrix = args
    return generate_random_bayes_net(x, num_nodes, max_in_degree, mi_matrix)

def generate_structured_bayes_net(args):
    x, num_nodes, max_in_degree, bootstrap = args
    x_ = x[np.random.choice(x.shape[0], x.shape[0], replace=True)] if bootstrap else x
    mi_matrix = mutual_information(x_)
    return generate_random_bayes_net(x, num_nodes, max_in_degree, mi_matrix=mi_matrix)

def compute_bayesian_network_score(bayes_net: BayesNetwork):
    return (bayes_net.copy(), bayes_net.baysian_score())

class GeneticSearch:
    def __init__(self, x, population_size, max_in_degree=4, mi_constraints=None):
        self.x = x
        self.num_nodes = x.shape[1]
        self.max_in_degree = max_in_degree
        self.population_size = population_size
        self.population = [] # (bayes_net, y) # bayes net, bayes score pair
        self.mi_matrix = mutual_information(x, mi_constraints)
        self.bayes_score_cache = {} # "shared cache"
    
    def init_population(self, structured_ratio=0.75, bootstrap=True, init_pop:List[BayesNetwork] = None):
        # structured_ratio: fraction of candidates generated with mutual information ranking
        logging.info(f"Generating initial population with structured ratio: {structured_ratio}")
        
        if init_pop is not None:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                pop = list(executor.map(compute_bayesian_network_score, args_list))
            self.population.extend(pop)

        cnt_random_candidates = int((1 - structured_ratio) * (self.population_size - len(self.population)))
        cnt_structured_candidates = self.population_size - cnt_random_candidates

        num_cores = multiprocessing.cpu_count()
        # x, num_nodes, max_in_degree, mi_matrix = args
        args_list = [(self.x, self.num_nodes, self.max_in_degree, None) for _ in range(cnt_random_candidates)]
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            random_bayes_nets = list(executor.map(generate_random_bayes_net_wrapper, args_list))
        
        self.population.extend(random_bayes_nets)
        
        args_list = [(self.x, self.num_nodes, self.max_in_degree, bootstrap) for _ in range(cnt_structured_candidates)]
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            structured_dags = list(executor.map(generate_structured_bayes_net, args_list))
        
        self.population.extend(structured_dags)

    def select_candidate(self):
        fitness_scores = [x[1] for x in self.population]

        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        return random.choices(self.population, weights=selection_probs, k=1)[0]
    
    def next_gen(self, mutation_rate=0.025, elite_ratio=0.1):
        num_cores = multiprocessing.cpu_count()

        # update and share bayesian score component cache across population
        for x in self.population:
            self.bayes_score_cache.update(x[0].bayesian_score_cache) # shared cache      

        self.population.sort(key=lambda x: x[1], reverse=True) # sort by fitness

        elite_count = int(self.population_size * elite_ratio)
        elites = self.population[:elite_count]
        new_population = elites.copy()

        # Prepare arguments for generate_offspring
        offspring_count = self.population_size - elite_count
        args_list = [
            (
                self.select_candidate(),
                self.select_candidate(),
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

    def fit(self, n_generations):
        num_cores = multiprocessing.cpu_count()
        logging.info(f'Start training GeneticSearch, num_cores = {num_cores}')
        for i in range(n_generations):
            logging.info(f'Generation: {i+1}')
            self.next_gen()

            self.population.sort(key=lambda x: x[1], reverse=True) # only for logging
            logging.info(f'Top Candidate Bayesian Score: {self.population[0][1]}')
            logging.info(f'Top Candidate edges: {self.population[0][0].G.edges}')

            logging.info(f'Worst candidate Bayesian Score: {self.population[-1][1]}')
            logging.info(f'Worst candidate edges: {self.population[-1][0].G.edges}')
        
        logging.info(f'Traning GeneticSearch completed')

def dump_last_generation(genetic_search: GeneticSearch, filename):
    top_score = genetic_search.population[0][1]

    with open(f'pickles/{filename}_({(round(top_score, 2))}).pkl', 'wb') as f:
        pickle.dump(genetic_search.population, f)
    
    return genetic_search.population

def compute_genetic_search(x, population_size, bootstrap_init, structured_init_ratio, max_in_degree, n_generations=10, dumpfilename='dump', init_pop=None):
    logging.info(f"Running GeneticSearch\n population size: {population_size}, boostrap_init: {bootstrap_init}, structured_init_ratio: {structured_init_ratio}, max_in_degree: {max_in_degree}")
    genetic_search = GeneticSearch(x, population_size=population_size, max_in_degree=max_in_degree)
    genetic_search.init_population(structured_ratio=structured_init_ratio, bootstrap=bootstrap_init, init_pop=init_pop)
    genetic_search.fit(n_generations=n_generations)
    logging.info(f"Size of bayesian score cache: items, {len(genetic_search.bayes_score_cache.keys())}, {sys.getsizeof(genetic_search.bayes_score_cache) / (1024**2)}MB")
    return dump_last_generation(genetic_search, f"{dumpfilename}") # this returns candidates from last generation
